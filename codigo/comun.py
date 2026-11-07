# =============================================================================
# Lo usan todas.
# Pili: lectura (_ruta, leer_tramo, tramos) y filtrar_ecg
# Juani: pan_tompkins, emparejar
# Sofi: cambios_de_ritmo, etiqueta_en, subject_ids
# =============================================================================
# FUNCIONES COMUNES
#
# Lectura de registros por tramos, filtrado, Pan-Tompkins, emparejamiento de
# latidos y etiquetas de ritmo. Las usan todas las etapas.

import os
import sys
import csv
from collections import deque

import numpy as np
import wfdb
from scipy import signal

import config


# ---------------------------------------------------------------------------
# Lectura
# ---------------------------------------------------------------------------

def todos_los_registros():
    """Los registros anotados de la etapa 0, sin los que tienen marcapasos
    durante el Holter (ritmo marcapaseado = regular aunque haya FA)."""
    ruta = os.path.join(config.RES, "registros_anotados.csv")
    with open(ruta, encoding="utf-8-sig", newline="") as f:
        return [fila["Data_ID"] for fila in csv.DictReader(f)
                if fila["PPM_on_Holter"].strip().lower() != "true"]


def registros_a_procesar():
    """Registros pasados por linea de comandos (python x.py 001 002 o
    python x.py todos) o, si no se pasa ninguno, los de config.REGISTROS
    (que tambien puede ser "todos")."""
    pedidos = sys.argv[1:] or config.REGISTROS
    if pedidos == "todos" or pedidos == ["todos"]:
        return todos_los_registros()
    return list(pedidos)


def _ruta(reg):
    """Si el registro esta en DATA_DIR se lee de ahi; si no, de PhysioNet
    (wfdb lo trae por internet sin guardar el .dat)."""
    local = os.path.join(config.DATA_DIR, reg)
    if os.path.exists(local + ".hea"):
        return local, None
    return reg, config.PN_DIR


def largo_registro(reg):
    nombre, pn = _ruta(reg)
    return wfdb.rdheader(nombre, pn_dir=pn).sig_len


def leer_tramo(reg, a, b):
    """Devuelve las muestras [a, b) del canal elegido, en mV."""
    nombre, pn = _ruta(reg)
    rec = wfdb.rdrecord(nombre, sampfrom=a, sampto=b,
                        channels=[config.CANAL], pn_dir=pn)
    return rec.p_signal[:, 0]


def leer_anotaciones(reg, ext):
    nombre, pn = _ruta(reg)
    return wfdb.rdann(nombre, ext, pn_dir=pn)


def tramos(n_total):
    """Recorre el registro de a TRAMO_MIN minutos con solape.
    Devuelve (inicio_lectura, fin_lectura, inicio_util, fin_util).
    Lo detectado en el solape se descarta: solo se guarda lo de [inicio_util, fin_util)."""
    largo = int(config.TRAMO_MIN * 60 * config.FS)
    solape = int(config.SOLAPE_S * config.FS)
    for a in range(0, n_total, largo):
        b = min(a + largo, n_total)
        yield max(a - solape, 0), min(b + solape, n_total), a, b


# ---------------------------------------------------------------------------
# Etiquetas de ritmo (.atr)
# ---------------------------------------------------------------------------
# En el .atr cada latido tiene una anotacion; el ritmo aparece en aux_note
# solo cuando cambia ('(AFIB', '(N', '(AFL', ...). FA = '(AFIB'; no-FA = '(N'.
# Flutter, taquicardias auriculares, etc. quedan como 'otro' y se excluyen.

MAPA_RITMO = {"(AFIB": "FA", "(N": "noFA"}


def cambios_de_ritmo(reg):
    """Devuelve (muestras donde cambia el ritmo, etiqueta de cada tramo)."""
    atr = leer_anotaciones(reg, "atr")
    muestras, etiquetas = [], []
    for s, nota in zip(atr.sample, atr.aux_note):
        nota = nota.strip().strip("\x00")
        if nota:
            muestras.append(s)
            etiquetas.append(MAPA_RITMO.get(nota, "otro"))
    return np.array(muestras), etiquetas


def etiqueta_en(muestras, inicios, etiquetas):
    """Etiqueta de ritmo vigente en cada muestra pedida."""
    idx = np.searchsorted(inicios, muestras, side="right") - 1
    tabla = np.array(etiquetas + ["sin_etiqueta"], dtype=object)
    return tabla[np.where(idx >= 0, idx, len(etiquetas))]


def subject_ids():
    """Diccionario Data_ID -> Subject_ID (sale de la etapa 0)."""
    ruta = os.path.join(config.RES, "registros_anotados.csv")
    with open(ruta, encoding="utf-8-sig", newline="") as f:
        return {fila["Data_ID"]: fila["Subject_ID"] for fila in csv.DictReader(f)}


# ---------------------------------------------------------------------------
# Filtrado (etapa 3)
# ---------------------------------------------------------------------------

def filtrar_ecg(x, fs=config.FS):
    """Pasa-banda Butterworth con filtfilt (fase cero: no deforma el ST ni
    corre los latidos). Notch opcional, solo si la etapa 2 lo justifica."""
    sos = signal.butter(2, config.PASABANDA_HZ, btype="bandpass", fs=fs, output="sos")
    y = signal.sosfiltfilt(sos, x)
    if config.NOTCH_HZ:
        b, a = signal.iirnotch(config.NOTCH_HZ, Q=30, fs=fs)
        y = signal.filtfilt(b, a, y)
    return y


# ---------------------------------------------------------------------------
# Pan-Tompkins (etapa 5)
# ---------------------------------------------------------------------------

def pan_tompkins(x, fs=config.FS):
    """Detector de QRS: preprocesado de Pan y Tompkins (1985) + reglas de
    decision optimizadas de Hamilton y Tompkins (1986), version offline.

    Preprocesado (Pan y Tompkins 1985): pasa-banda 5-15 Hz, derivada de
    5 puntos, cuadrado e integracion de 150 ms. Diferencia con el original:
    todas las etapas son de fase cero (filtfilt, derivada y media movil
    centradas), asi que NO hay retardos que compensar. En la version causal
    habria que restar 6+16 (pasa-banda) + 2 (derivada) + ~15 (media movil)
    muestras a 200 Hz. Los filtros de coeficientes enteros del original se
    eligieron para correr en tiempo real en un microprocesador
    (Pan y Tompkins 1985; Hamilton y Tompkins 1986, p. 1158).

    Reglas de decision (Hamilton y Tompkins 1986, p. 1164, version final):
      - nivel de QRS y de ruido = MEDIANA de los ultimos 8 picos de cada
        tipo (fue el mejor estimador; el promedio iterativo fue el peor)
      - umbral = ruido + 0,1825 * (QRS - ruido)
      - periodo refractario de 200 ms
      - busqueda hacia atras: si no hubo QRS en 1,5 * mediana de los
        ultimos 8 RR, se toma el pico de ruido mas alto de ese intervalo
        que supere la mitad del umbral
      - onda T: entre 200 y 360 ms despues de un QRS, se rechaza si su
        derivada maxima es menor a la mitad de la del QRS anterior

    Devuelve las muestras de los picos R (dentro de x)."""
    # 1) Pasa-banda 5-15 Hz: realza el QRS, atenua P, T, deriva, EMG y red
    sos = signal.butter(2, [5, 15], btype="bandpass", fs=fs, output="sos")
    bp = signal.sosfiltfilt(sos, x)

    # 2) Derivada de 5 puntos (centrada): pendiente del QRS
    d = np.zeros_like(bp)
    d[2:-2] = (-bp[:-4] - 2 * bp[1:-3] + 2 * bp[3:-1] + bp[4:]) * fs / 8

    # 3) Cuadrado: todo positivo y realza pendientes grandes
    c = d ** 2

    # 4) Integracion por ventana movil de 150 ms
    n_mwi = int(round(0.150 * fs))
    mwi = np.convolve(c, np.ones(n_mwi) / n_mwi, mode="same")

    # 5) Reglas de decision de Hamilton y Tompkins (1986) sobre los picos de
    #    la senal integrada. Se toman maximos separados al menos 200 ms
    #    (un solo QRS cabe en 200 ms; evita picos multiples por ondulaciones).
    picos, _ = signal.find_peaks(mwi, distance=int(0.2 * fs))
    if len(picos) == 0:
        return np.array([], dtype=int)

    TC = 0.1825                    # coeficiente del umbral (p. 1164)
    refractario = int(0.200 * fs)  # 200 ms (p. 1162)
    ventana_t = int(0.360 * fs)    # onda T: 200-360 ms (p. 1164)
    w_pend = int(0.075 * fs)       # ventana para la derivada maxima del QRS

    # Inicializacion: el nivel de QRS arranca con el maximo de cada uno de
    # los primeros 8 s (aprox. un latido por segundo) y el de ruido en 0.
    n_ini = max(min(8, len(mwi) // fs), 1)
    pk_qrs = deque([mwi[i * fs:(i + 1) * fs].max() for i in range(n_ini)], maxlen=8)
    pk_ruido = deque([0.0] * 8, maxlen=8)
    rr = deque([fs] * 8, maxlen=8)   # RR inicial de 1 s

    def umbral():
        nivel_ruido = np.median(pk_ruido)
        nivel_qrs = np.median(pk_qrs)
        return nivel_ruido + TC * (nivel_qrs - nivel_ruido)

    # Derivada del ECG (no del filtrado 5-15 Hz) para distinguir ondas T
    dx = np.abs(np.gradient(x))

    def pendiente(i):
        return dx[max(i - w_pend, 0):i + w_pend + 1].max()

    qrs, ruido = [], []

    def es_qrs(i):
        if qrs:
            rr.append(i - qrs[-1])
        qrs.append(i)
        pk_qrs.append(mwi[i])

    for p in picos:
        if qrs and p - qrs[-1] < refractario:
            continue

        # Busqueda hacia atras
        if qrs and p - qrs[-1] > 1.5 * np.median(rr):
            mitad = 0.5 * umbral()
            cand = [q for q in reversed(ruido) if q > qrs[-1] + refractario]
            cand = [q for q in cand if mwi[q] > mitad]
            if cand:
                es_qrs(max(cand, key=lambda q: mwi[q]))

        v = mwi[p]
        if v > umbral():
            # Posible onda T
            if qrs and p - qrs[-1] < ventana_t and pendiente(p) < 0.5 * pendiente(qrs[-1]):
                ruido.append(p)
                pk_ruido.append(v)
                continue
            es_qrs(p)
        else:
            ruido.append(p)
            pk_ruido.append(v)
            if len(ruido) > 64:          # solo hacen falta los recientes
                del ruido[:32]

    # 6) Ubicar la R: maximo de |ECG| en +-100 ms alrededor de cada deteccion
    #    (Hamilton y Tompkins usan el pico mas grande de la senal pasa-banda
    #    en una ventana de 100 ms antes de la deteccion causal; aca la senal
    #    integrada esta centrada, asi que la ventana se centra en el pico)
    w = int(0.1 * fs)
    r = []
    for q in sorted(qrs):
        a, b = max(q - w, 0), min(q + w, len(x))
        seg = x[a:b]
        r.append(a + int(np.argmax(np.abs(seg - np.median(seg)))))
    r = np.unique(r)
    # si dos R quedaron a menos de 200 ms, se deja la primera
    if len(r) > 1:
        r = r[np.insert(np.diff(r) >= 0.2 * fs, 0, True)]
    return r


# ---------------------------------------------------------------------------
# Comparacion de detecciones
# ---------------------------------------------------------------------------

def emparejar(ref, det, tol):
    """Empareja latidos de referencia y detectados (ambos ordenados) con
    tolerancia tol (en muestras). Devuelve (VP, FN, FP)."""
    i = j = vp = 0
    while i < len(ref) and j < len(det):
        dif = det[j] - ref[i]
        if abs(dif) <= tol:
            vp += 1
            i += 1
            j += 1
        elif dif < 0:
            j += 1
        else:
            i += 1
    return vp, len(ref) - vp, len(det) - vp


def guardar_csv(filas, ruta, columnas):
    os.makedirs(os.path.dirname(ruta), exist_ok=True)
    with open(ruta, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=columnas)
        w.writeheader()
        w.writerows(filas)


def actualizar_csv(filas, ruta, columnas, clave="registro"):
    """Como guardar_csv, pero conserva las filas de otros registros que ya
    estaban en el archivo. Asi se puede correr registro por registro (o en
    paralelo) sin que cada corrida pise los resultados de las anteriores."""
    nuevos = {f[clave] for f in filas}
    viejas = []
    if os.path.exists(ruta):
        viejas = [f for f in leer_csv(ruta) if f[clave] not in nuevos]
    todas = sorted(viejas + filas, key=lambda f: str(f[clave]))
    guardar_csv(todas, ruta, columnas)


def leer_csv(ruta):
    with open(ruta, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def leer_caracteristicas():
    """Ventanas de resultados/caracteristicas.csv sin los registros de
    config.REGISTROS_EXCLUIDOS. Usarla en la estadistica y en la ROC para que
    todas trabajen con el mismo conjunto."""
    filas = leer_csv(os.path.join(config.RES, "caracteristicas.csv"))
    return [f for f in filas if f["registro"] not in config.REGISTROS_EXCLUIDOS]
