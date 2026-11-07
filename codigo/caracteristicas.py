# =============================================================================
# RESPONSABLES:
#   Sofi  -> ventanas_registro() (etapa 7)
#   Juani -> caracteristicas temporales e irregularidad: tpr, shannon, sampen,
#            media_rr, sdnn, rmssd, nrmssd (etapa 8)
#   Pili  -> lomb_bandas() y psd_ar() (etapa 8, parte espectral)
# Hecho (Sofi, avance):
#   [x] Comparar VENTANA_LATIDOS = 60 y 128: ventanas por clase y boxplots.
# Hecho (Juani, avance):
#   [x] Comparar medianas de nrmssd, tpr y shannon con Dash 2009
#       (> 0,1; 0,54-0,77; > 0,7).
# Hecho (Pili, TP completo):
#   [x] Implementar psd_ar() (hecho). Comparacion con Lomb: python codigo/comparar_ar_lomb.py
# =============================================================================
# ETAPAS 7 y 8: ventanas y caracteristicas
#
# Que hace: corta la serie RR de cada registro en ventanas de VENTANA_LATIDOS
# intervalos consecutivos (sin solape). Se queda con las ventanas que:
#   - tienen una sola etiqueta de ritmo (FA o noFA)
#   - tienen al menos MIN_FRACCION_VALIDA de intervalos validos
# y calcula, sobre los intervalos validos de cada ventana:
#   Temporales:   media_rr, sdnn, rmssd (Task Force 1996)
#   Irregularidad: nrmssd = rmssd/media, tpr, shannon (Dash et al. 2009),
#                  sampen (entropia muestral, m=2, r=0,2*SD)
#   Espectrales:  lf, hf (potencia relativa), lf_hf con periodograma de Lomb
#                 (Laguna et al. 1998).
#                 OJO: con ventanas de ~1 min LF no es confiable (el Task
#                 Force recomienda >= 2 min para LF). Se reporta igual para
#                 mostrarlo, pero hay que aclararlo en el informe.
#
# Entra:  resultados/rr/<reg>.csv, resultados/registros_anotados.csv
# Sale:   resultados/caracteristicas.csv  (una fila por ventana)
#         resultados/figuras/boxplots_caracteristicas.png
#
# Como correrlo:
#   python codigo/caracteristicas.py [registros]

import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import signal
from scipy.integrate import trapezoid

import config
import comun


# ---------------------------------------------------------------------------
# Caracteristicas
# ---------------------------------------------------------------------------

def tpr(x):
    """Turning point ratio: fraccion de puntos que son maximo o minimo local.
    Para una serie aleatoria vale ~2/3 (Dash et al. 2009)."""
    c = x[1:-1]
    tp = np.sum(((c > x[:-2]) & (c > x[2:])) | ((c < x[:-2]) & (c < x[2:])))
    return tp / len(x)


def shannon(x, nbins=16, recorte=8):
    """Entropia de Shannon del histograma de RR, normalizada a [0, 1].
    Se sacan los 8 RR mas largos y los 8 mas cortos (Dash et al. 2009)."""
    s = np.sort(x)
    if len(s) > 2 * recorte + nbins:
        s = s[recorte:-recorte]
    cnt, _ = np.histogram(s, bins=nbins)
    p = cnt[cnt > 0] / len(s)
    return -np.sum(p * np.log(p)) / np.log(nbins)


def sampen(x, m=2, r=0.2):
    """Entropia muestral (Richman y Moorman 2000). r en unidades de SD."""
    x = np.asarray(x, dtype=float)
    tol = r * np.std(x)
    if tol == 0 or len(x) < m + 2:
        return np.nan

    def pares(mm):
        e = np.lib.stride_tricks.sliding_window_view(x, mm)[:len(x) - m]
        d = np.max(np.abs(e[:, None, :] - e[None, :, :]), axis=2)
        return (np.sum(d <= tol) - len(e)) / 2

    b, a = pares(m), pares(m + 1)
    return -np.log(a / b) if a > 0 and b > 0 else np.nan


def lomb_bandas(t, rr):
    """Potencia relativa en LF (0,04-0,15 Hz) y HF (0,15-0,4 Hz) con Lomb."""
    f = np.linspace(0.01, 0.5, 250)
    p = signal.lombscargle(t, rr - rr.mean(), 2 * np.pi * f)
    total = trapezoid(p, f)
    lf = trapezoid(p[(f >= 0.04) & (f < 0.15)], f[(f >= 0.04) & (f < 0.15)])
    hf = trapezoid(p[(f >= 0.15) & (f < 0.40)], f[(f >= 0.15) & (f < 0.40)])
    return lf / total, hf / total, (lf / hf if hf > 0 else np.nan)



def levinson_durbin(r, orden_max):
    """Levinson-Durbin sobre la autocorrelacion r[0..orden_max].
    Devuelve, para cada orden p = 1..orden_max, los coeficientes a (con a[0] = 1)
    y la varianza del error de prediccion sigma2_p."""
    a = np.array([1.0])
    err = r[0]
    modelos = []
    for p in range(1, orden_max + 1):
        k = -(r[p] + np.dot(a[1:], r[p - 1:0:-1])) / err    # coeficiente de reflexion
        a = np.concatenate([a, [0.0]])
        a = a + k * a[::-1]
        err = err * (1 - k ** 2)
        modelos.append((a.copy(), err))
    return modelos


def psd_ar(t, rr, orden_max=20, fs_re=4.0):
    """ETAPA 8 (Pili) - PSD parametrica AR para comparar con lomb_bandas().

      1. remuestrea la serie RR a fs_re = 4 Hz (interpolacion lineal), le resta la media
      2. estima la autocorrelacion y los coeficientes AR por Yule-Walker
         (Levinson-Durbin) para ordenes 1..orden_max
      3. elige el orden con AIC = N ln(sigma2) + 2p (Boardman et al. 2002)
      4. calcula la PSD del modelo y la potencia relativa en LF y HF
         (mismas bandas y mismo rango 0,01-0,5 Hz que lomb_bandas)
    Devuelve (lf, hf, lf_hf, orden_elegido). Si la ventana es muy corta o
    constante, devuelve (nan, nan, nan, 0).
    """
    t = np.asarray(t, dtype=float)
    rr = np.asarray(rr, dtype=float)
    tt = np.arange(t[0], t[-1], 1 / fs_re)
    if len(tt) < 4 * orden_max:
        return np.nan, np.nan, np.nan, 0
    x = np.interp(tt, t, rr)
    x = x - x.mean()
    N = len(x)
    if np.allclose(x, 0):
        return np.nan, np.nan, np.nan, 0

    # autocorrelacion sesgada (garantiza un modelo estable)
    r = np.array([np.dot(x[:N - k], x[k:]) / N for k in range(orden_max + 1)])
    modelos = levinson_durbin(r, orden_max)
    aic = [N * np.log(err) + 2 * p for p, (_, err) in enumerate(modelos, start=1)]
    p_opt = int(np.argmin(aic)) + 1
    a, sigma2 = modelos[p_opt - 1]

    f = np.linspace(0.01, 0.5, 250)
    z = np.exp(-2j * np.pi * f[:, None] * np.arange(len(a))[None, :] / fs_re)
    psd = sigma2 / fs_re / np.abs(z @ a) ** 2
    total = trapezoid(psd, f)
    lf = trapezoid(psd[(f >= 0.04) & (f < 0.15)], f[(f >= 0.04) & (f < 0.15)])
    hf = trapezoid(psd[(f >= 0.15) & (f < 0.40)], f[(f >= 0.15) & (f < 0.40)])
    return lf / total, hf / total, (lf / hf if hf > 0 else np.nan), p_opt


def caracteristicas(t, rr):
    media = rr.mean()
    rmssd = np.sqrt(np.mean(np.diff(rr) ** 2))
    lf, hf, lf_hf = lomb_bandas(t, rr)
    return {
        "media_rr": media,
        "sdnn": rr.std(ddof=1),
        "rmssd": rmssd,
        "nrmssd": rmssd / media,
        "tpr": tpr(rr),
        "shannon": shannon(rr),
        "sampen": sampen(rr),
        "lf": lf, "hf": hf, "lf_hf": lf_hf,
    }


NOMBRES = ["media_rr", "sdnn", "rmssd", "nrmssd", "tpr", "shannon",
           "sampen", "lf", "hf", "lf_hf"]


# ---------------------------------------------------------------------------
# Ventanas
# ---------------------------------------------------------------------------

def ventanas_registro(reg, sujetos):
    filas = comun.leer_csv(os.path.join(config.RES, "rr", f"{reg}.csv"))
    t = np.array([float(f["t_s"]) for f in filas])
    rr = np.array([float(f["rr_s"]) for f in filas])
    et = np.array([f["etiqueta"] for f in filas])
    ok = np.array([f["valido"] == "1" for f in filas])

    n = config.VENTANA_LATIDOS
    salida = []
    for i in range(0, len(rr) - n + 1, n):
        sl = slice(i, i + n)
        etiquetas = set(et[sl])
        if len(etiquetas) != 1 or etiquetas.pop() not in ("FA", "noFA"):
            continue
        if ok[sl].mean() < config.MIN_FRACCION_VALIDA:
            continue
        m = ok[sl]
        fila = {"registro": reg, "sujeto": sujetos.get(reg, ""),
                "t_inicio_s": round(float(t[i]), 1), "etiqueta": et[i],
                "n_validos": int(m.sum())}
        fila.update({k: round(float(v), 5)
                     for k, v in caracteristicas(t[sl][m], rr[sl][m]).items()})
        salida.append(fila)

    n_fa = sum(f["etiqueta"] == "FA" for f in salida)
    print(f"  {reg}: {len(salida)} ventanas ({n_fa} FA, {len(salida) - n_fa} noFA)")
    return salida


def boxplots(filas):
    et = np.array([f["etiqueta"] for f in filas])
    fig, axs = plt.subplots(2, 5, figsize=(15, 6))
    for ax, nombre in zip(axs.flat, NOMBRES):
        v = np.array([f[nombre] for f in filas], dtype=float)
        datos = [v[(et == e) & np.isfinite(v)] for e in ("noFA", "FA")]
        ax.boxplot(datos, showfliers=False)
        ax.set_xticks([1, 2], ["noFA", "FA"])
        ax.set_title(nombre)
        ax.grid(alpha=0.3)
    n_fa = int(np.sum(et == "FA"))
    fig.suptitle(f"Caracteristicas por ventana de {config.VENTANA_LATIDOS} latidos "
                 f"({n_fa} FA, {len(et) - n_fa} noFA)")
    fig.tight_layout()
    fig.savefig(os.path.join(config.FIG, "boxplots_caracteristicas.png"), dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    os.makedirs(config.FIG, exist_ok=True)
    sujetos = comun.subject_ids()
    filas = []
    for reg in comun.registros_a_procesar():
        if not os.path.exists(os.path.join(config.RES, "rr", f"{reg}.csv")):
            print(f"  {reg}: falta resultados/rr/{reg}.csv (correr serie_rr.py), se saltea")
            continue
        filas += ventanas_registro(reg, sujetos)

    columnas = ["registro", "sujeto", "t_inicio_s", "etiqueta", "n_validos"] + NOMBRES
    comun.guardar_csv(filas, os.path.join(config.RES, "caracteristicas.csv"), columnas)
    # el duplicado queda en el CSV pero no entra en la figura ni en las medianas
    filas = [f for f in filas if f["registro"] not in config.REGISTROS_EXCLUIDOS]
    boxplots(filas)

    # Medianas por clase (para una primera mirada; la estadistica es la etapa 9)
    et = np.array([f["etiqueta"] for f in filas])
    print(f"\n{'':10s}{'noFA':>10s}{'FA':>10s}")
    for nombre in NOMBRES:
        v = np.array([f[nombre] for f in filas], dtype=float)
        print(f"{nombre:10s}" + "".join(f"{np.nanmedian(v[et == e]):10.4f}"
                                        for e in ("noFA", "FA")))
