# =============================================================================
# RESPONSABLE: Pili
# Hecho (Pili, avance):
#   [x] Revisar figuras ruido_<reg>.png de varios registros. Si marca ventanas
#       limpias como ruidosas (o al reves), ajustar UMBRAL_BSQI / UMBRAL_KSQI
#       en config.py y documentar el criterio.
# Hecho (Pili, TP completo):
#   [x] Etapa 4: ale_nlms() implementada.
#   [x] Etapa 4: evaluar_ale.py corrido (128 combinaciones); decision: no se integra.
# =============================================================================
# ETAPA 3: filtrado y deteccion de ruido
#
# Que hace:
#   - filtra cada tramo con el pasa-banda de config (y notch si se activo)
#   - divide el registro en ventanas de 10 s y calcula dos indices de calidad:
#       bSQI: acuerdo entre dos detectores de QRS distintos (Pan-Tompkins y
#             xqrs de wfdb). Si el ECG esta limpio, coinciden; si hay ruido
#             muscular o movimiento, no (Li, Mark y Clifford 2008).
#       kSQI: curtosis del ECG. Un ECG limpio es "picudo" (curtosis > 5);
#             el ruido tiende a una distribucion mas gaussiana.
#   - marca como ruidosa la ventana segun config.CRITERIO_RUIDO (por defecto, bSQI < UMBRAL_BSQI)

# El filtrado adaptativo (ALE/NLMS, etapa 4) esta en ale_nlms(). Se evalua
# sobre ventanas con bSQI bajo con evaluar_ale.py (resultados/evaluacion_ale.csv).
# En la evaluacion no mejoro la deteccion de QRS, asi que el pipeline no lo
# aplica: las ventanas ruidosas se excluyen.
#
# Entra:  registros
# Sale:   resultados/calidad/<reg>.csv  (una fila por ventana de 10 s)
#         resultados/figuras/filtrado_<reg>.png
#         resultados/figuras/ruido_<reg>.png
#
# Como correrlo:
#   python codigo/preprocesado.py [registros]

import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import kurtosis
from wfdb import processing

import config
import comun

def es_ruidosa(bsqi, ksqi):
    """Decide si una ventana de 10 s es ruidosa segun config.CRITERIO_RUIDO."""
    if config.CRITERIO_RUIDO == "bsqi":
        return bsqi < config.UMBRAL_BSQI
    if config.CRITERIO_RUIDO == "bsqi_ksqi":
        return bsqi < config.UMBRAL_BSQI or ksqi < config.UMBRAL_KSQI
    raise ValueError(f"CRITERIO_RUIDO desconocido: {config.CRITERIO_RUIDO}")

def calidad_registro(reg):
    fs = config.FS
    n_total = comun.largo_registro(reg)
    tol = int(config.TOL_QRS_S * fs)
    largo_v = int(config.VENTANA_CALIDAD_S * fs)
    filas = []

    for a_lec, b_lec, a, b in comun.tramos(n_total):
        x = comun.filtrar_ecg(comun.leer_tramo(reg, a_lec, b_lec))
        r_pt = comun.pan_tompkins(x) + a_lec
        r_xq = processing.xqrs_detect(x, fs=fs, verbose=False) + a_lec

        for v0 in range(a, b, largo_v):
            v1 = min(v0 + largo_v, b)
            if v1 - v0 < largo_v // 2:
                continue
            p = r_pt[(r_pt >= v0) & (r_pt < v1)]
            q = r_xq[(r_xq >= v0) & (r_xq < v1)]
            vp, fn, fp = comun.emparejar(q, p, tol)
            bsqi = vp / (vp + fn + fp) if (vp + fn + fp) else 0.0
            ksqi = kurtosis(x[v0 - a_lec:v1 - a_lec], fisher=False)
            filas.append({
                "inicio_muestra": v0, "fin_muestra": v1,
                "bsqi": round(bsqi, 3), "ksqi": round(float(ksqi), 2),
                "ruidosa": int(es_ruidosa(bsqi, ksqi)),
            })
        print(f"  {reg}: {b / fs / 3600:5.1f} h procesadas", end="\r")

    ruta = os.path.join(config.RES, "calidad", f"{reg}.csv")
    comun.guardar_csv(filas, ruta, list(filas[0].keys()))
    n_r = sum(f["ruidosa"] for f in filas)
    print(f"\n  {reg}: {n_r}/{len(filas)} ventanas ruidosas "
          f"({100 * n_r / len(filas):.1f}%) -> {ruta}")
    return filas



def ale_nlms(x, delta=10, L=64, mu=0.01, eps=1e-8):
    """ETAPA 4 (Pili) - Mejorador adaptativo de linea (ALE) con NLMS.

    Widrow et al. 1975; Fotiadou et al. 2018. Se usa la propia senal
    retrasada delta muestras como referencia:
      r[n] = x[n - delta]                    (vector de las L muestras previas)
      y[n] = w[n] . r[n]                     (parte correlacionada = ECG mejorado)
      e[n] = x[n] - y[n]                     (lo que no se puede predecir = ruido)
      w[n+1] = w[n] + mu * e[n] * r[n] / (eps + ||r[n]||^2)    (NLMS)
    El ECG se repite latido a latido (correlacion larga); el ruido muscular se
    decorrela en pocas muestras. Si delta es mayor que el ancho de la
    autocorrelacion del ruido, el filtro solo puede predecir el ECG.

    Parametros:
      delta: retardo en muestras (> ancho de autocorrelacion del ruido)
      L:     cantidad de coeficientes (20 a 320 a 200 Hz)
      mu:    paso, 0 < mu < 2
    Devuelve (y, e): senal mejorada y error. Los primeros delta + L
    muestras son el transitorio de convergencia: no usarlos para medir.
    """
    x = np.asarray(x, dtype=float)
    n = len(x)
    w = np.zeros(L)
    y = np.zeros(n)
    e = np.zeros(n)
    for i in range(delta + L, n):
        r = x[i - delta - L + 1:i - delta + 1][::-1]       # L muestras, la mas reciente primero
        y[i] = w @ r
        e[i] = x[i] - y[i]
        w += mu * e[i] * r / (eps + r @ r)
    return y, e


def figura_filtrado(reg, t0_s=3600):
    """ECG crudo vs filtrado, 10 s."""
    fs = config.FS
    a = int(t0_s * fs)
    x = comun.leer_tramo(reg, a - 10 * fs, a + 20 * fs)   # margen para filtfilt
    y = comun.filtrar_ecg(x)
    t = (np.arange(len(x)) + a - 10 * fs) / fs
    sl = slice(10 * fs, 20 * fs)

    fig, ax = plt.subplots(2, 1, figsize=(11, 5), sharex=True)
    ax[0].plot(t[sl], x[sl], lw=0.8)
    ax[0].set_title(f"Registro {reg} – crudo")
    ax[1].plot(t[sl], y[sl], lw=0.8, color="C1")
    ax[1].set_title(f"Filtrado: pasa-banda {config.PASABANDA_HZ} Hz (filtfilt)"
                    + (f" + notch {config.NOTCH_HZ} Hz" if config.NOTCH_HZ else ""))
    for a_ in ax:
        a_.set_ylabel("mV")
        a_.grid(alpha=0.3)
    ax[1].set_xlabel("Tiempo [s]")
    fig.tight_layout()
    fig.savefig(os.path.join(config.FIG, f"filtrado_{reg}.png"), dpi=150)
    plt.close(fig)


def figura_ruido(reg, filas):
    """Ventana tipica (bSQI = 1, curtosis mediana) vs la ventana con peor
    bSQI, con las dos detecciones."""
    fs = config.FS
    k_med = np.median([f["ksqi"] for f in filas])
    tipica = min((f for f in filas if f["bsqi"] == max(g["bsqi"] for g in filas)),
                 key=lambda f: abs(f["ksqi"] - k_med))
    peor = min(filas, key=lambda f: (f["bsqi"], f["ksqi"]))
    margen = 60 * fs   # se detecta con 1 min de contexto, como en el registro completo

    fig, axs = plt.subplots(2, 1, figsize=(11, 5))
    for ax, fila, txt in [(axs[0], tipica, "tipica"), (axs[1], peor, "con peor bSQI")]:
        v0, v1 = fila["inicio_muestra"], fila["fin_muestra"]
        a = max(v0 - margen, 0)
        fin = min(v1 + margen, comun.largo_registro(reg))
        x = comun.filtrar_ecg(comun.leer_tramo(reg, a, fin))
        r_pt = comun.pan_tompkins(x) + a
        r_xq = processing.xqrs_detect(x, fs=fs, verbose=False) + a
        seg = x[v0 - a:v1 - a]
        t = np.arange(v0, v1) / fs
        r_pt = r_pt[(r_pt >= v0) & (r_pt < v1)] - v0
        r_xq = r_xq[(r_xq >= v0) & (r_xq < v1)] - v0
        ax.plot(t, seg, lw=0.8)
        ax.plot(t[r_pt], seg[r_pt], "rv", label="Pan-Tompkins")
        ax.plot(t[r_xq], seg[r_xq] * 1.1, "g^", label="xqrs")
        ax.set_title(f"Ventana {txt}: bSQI = {fila['bsqi']}, kSQI = {fila['ksqi']}")
        ax.set_ylabel("mV")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, loc="upper right")
    axs[1].set_xlabel("Tiempo [s]")
    fig.tight_layout()
    fig.savefig(os.path.join(config.FIG, f"ruido_{reg}.png"), dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    os.makedirs(config.FIG, exist_ok=True)
    for reg in comun.registros_a_procesar():
        filas = calidad_registro(reg)
        figura_filtrado(reg)
        figura_ruido(reg, filas)
