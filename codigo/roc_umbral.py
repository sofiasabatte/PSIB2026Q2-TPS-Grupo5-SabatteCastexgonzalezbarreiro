# =============================================================================
# RESPONSABLE: Juani
# ETAPA 9b: curva ROC, umbral train/test y F1 de la deteccion de FA
#
# Que hace:
#   1. Separa las ventanas en train y test POR SUJETO (de Chazal et al. 2004):
#      todas las ventanas de un sujeto quedan del mismo lado.
#   2. Para cada caracteristica, en TRAIN: curva ROC, AUC y umbral que
#      maximiza el indice de Youden (Se + Esp - 1).
#   3. Aplica ese umbral en TEST (que el metodo nunca vio) y calcula
#      sensibilidad, especificidad, VPP, F1 y exactitud.
#   4. Evalua tambien la regla combinada de Dash et al. 2009, con sus
#      umbrales publicados (no se ajusta nada): FA si
#      nrmssd > 0,1  Y  0,54 <= tpr <= 0,77  Y  shannon > 0,7.
#   5. Estabilidad: repite 1-4 con N_PARTICIONES particiones distintas y
#      reporta media, desvio, minimo y maximo del F1 en test. Con ~97 sujetos
#      el resultado de UNA particion depende mucho de que sujetos caen en test.
#
# FA = clase positiva. F1 = 2 * VPP * Se / (VPP + Se).
#
# Diferencia con estadistica.py: alla el AUC es POR SUJETO (medianas) y sirve
# para decir si las clases difieren. Aca la unidad es la VENTANA, que es lo
# que el detector clasifica: mide que tan bien detecta la FA ventana a ventana.
#
# Ojo: Dash et al. fijaron sus umbrales con ventanas de 128 latidos; aca las
# ventanas son de config.VENTANA_LATIDOS (60).
#
# Entra:  resultados/caracteristicas.csv (sin config.REGISTROS_EXCLUIDOS)
# Sale:   resultados/roc_umbral.csv            (particion SEMILLA)
#         resultados/roc_umbral_estabilidad.csv (N_PARTICIONES particiones)
#         resultados/figuras/roc.png
#
# Como correrlo:
#   python codigo/roc_umbral.py
# =============================================================================

import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import rankdata

import config
import comun
from caracteristicas import NOMBRES

FRACCION_TEST = 0.3
SEMILLA = 0            # fija, para que la particion sea reproducible
N_PARTICIONES = 10
REGLA_DASH = "regla Dash 2009"


def particion_por_sujeto(filas, fraccion_test=FRACCION_TEST, semilla=SEMILLA):
    """Separa las ventanas en train y test sin repartir sujetos."""
    sujetos = sorted({f["sujeto"] for f in filas})
    rng = np.random.default_rng(semilla)
    rng.shuffle(sujetos)
    test = set(sujetos[:int(round(fraccion_test * len(sujetos)))])
    return ([f for f in filas if f["sujeto"] not in test],
            [f for f in filas if f["sujeto"] in test])


def a_arrays(filas, nombre):
    x = np.array([float(f[nombre]) for f in filas])
    y = np.array([f["etiqueta"] == "FA" for f in filas])
    ok = np.isfinite(x)
    return x[ok], y[ok]


def auc(x, y):
    """AUC por la formula de Mann-Whitney (probabilidad de que una ventana
    de FA tenga un valor mayor que una de no-FA)."""
    r = rankdata(x)
    n1, n0 = y.sum(), (~y).sum()
    return (r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def curva_roc(x, y):
    """Devuelve (umbrales, Se, Esp) recorriendo umbrales 'FA si x >= u'."""
    umbrales = np.unique(np.quantile(x, np.linspace(0, 1, 501)))
    se = np.array([(x[y] >= u).mean() for u in umbrales])
    esp = np.array([(x[~y] < u).mean() for u in umbrales])
    return umbrales, se, esp


def metricas(pred, y):
    vp = np.sum(pred & y); fn = np.sum(~pred & y)
    fp = np.sum(pred & ~y); vn = np.sum(~pred & ~y)
    se = vp / (vp + fn) if vp + fn else np.nan
    esp = vn / (vn + fp) if vn + fp else np.nan
    vpp = vp / (vp + fp) if vp + fp else np.nan
    f1 = 2 * vpp * se / (vpp + se) if (vpp + se) else np.nan
    return {"Se": se, "Esp": esp, "VPP": vpp, "F1": f1,
            "Exactitud": (vp + vn) / len(y)}


def regla_dash(filas):
    nr = np.array([float(f["nrmssd"]) for f in filas])
    tp = np.array([float(f["tpr"]) for f in filas])
    sh = np.array([float(f["shannon"]) for f in filas])
    y = np.array([f["etiqueta"] == "FA" for f in filas])
    return (nr > 0.1) & (tp >= 0.54) & (tp <= 0.77) & (sh > 0.7), y


def evaluar(train, test):
    """Umbral de Youden en train, metricas en test, para cada caracteristica
    y para la regla de Dash. Devuelve (filas de resultados, curvas ROC)."""
    salida, curvas = [], {}
    for nombre in NOMBRES:
        x_tr, y_tr = a_arrays(train, nombre)
        # sentido: si en FA la caracteristica es menor, se invierte el signo
        signo = 1 if np.median(x_tr[y_tr]) >= np.median(x_tr[~y_tr]) else -1
        x_tr = signo * x_tr
        umbrales, se, esp = curva_roc(x_tr, y_tr)
        u = umbrales[np.argmax(se + esp - 1)]          # indice de Youden
        x_te, y_te = a_arrays(test, nombre)
        salida.append({"caracteristica": nombre,
                       "AUC_train": auc(x_tr, y_tr),
                       "umbral": signo * u,
                       "FA_si": ">=" if signo == 1 else "<=",
                       **metricas(signo * x_te >= u, y_te)})
        curvas[nombre] = (1 - esp, se)
    pred, y = regla_dash(test)
    salida.append({"caracteristica": REGLA_DASH, "AUC_train": np.nan,
                   "umbral": np.nan, "FA_si": "fijo (publicado)",
                   **metricas(pred, y)})
    return salida, curvas


if __name__ == "__main__":
    filas = comun.leer_caracteristicas()
    train, test = particion_por_sujeto(filas)
    for nombre, parte in [("train", train), ("test", test)]:
        n_fa = sum(f["etiqueta"] == "FA" for f in parte)
        print(f"{nombre}: {len({f['sujeto'] for f in parte})} sujetos, "
              f"{len(parte)} ventanas ({n_fa} FA)")

    # --- 1. particion fija (SEMILLA) -----------------------------------------
    salida, curvas = evaluar(train, test)
    redondear = lambda v: round(float(v), 4) if isinstance(v, (float, np.floating)) else v
    comun.guardar_csv([{k: redondear(v) for k, v in s.items()} for s in salida],
                      os.path.join(config.RES, "roc_umbral.csv"), list(salida[0].keys()))

    fig, ax = plt.subplots(figsize=(6, 6))
    for s in salida[:-1]:
        fpr, tpr = curvas[s["caracteristica"]]
        ax.plot(fpr, tpr, lw=1, label=f"{s['caracteristica']} (AUC {s['AUC_train']:.2f})")
    ax.plot([0, 1], [0, 1], "k--", lw=0.8)
    ax.set_xlabel("1 - Especificidad")
    ax.set_ylabel("Sensibilidad")
    ax.set_title("Curvas ROC por caracteristica (train, por ventana)")
    ax.legend(fontsize=7, loc="lower right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    os.makedirs(config.FIG, exist_ok=True)
    fig.savefig(os.path.join(config.FIG, "roc.png"), dpi=150)

    print(f"\nParticion {SEMILLA} (Se, Esp, VPP y F1 en TEST; AUC en TRAIN)")
    print(f"{'caracteristica':18s}{'AUC':>7s}{'umbral':>9s}{'Se':>7s}{'Esp':>7s}{'VPP':>7s}{'F1':>7s}")
    for s in sorted(salida, key=lambda s: -s["F1"]):
        u = "fijo" if s["caracteristica"] == REGLA_DASH else f"{s['FA_si']}{s['umbral']:.3g}"
        a = "" if np.isnan(s["AUC_train"]) else f"{s['AUC_train']:.3f}"
        print(f"{s['caracteristica']:18s}{a:>7s}{u:>9s}{s['Se']:7.3f}{s['Esp']:7.3f}"
              f"{s['VPP']:7.3f}{s['F1']:7.3f}")

    # --- 2. estabilidad: N_PARTICIONES particiones ---------------------------
    f1s = {}
    for semilla in range(N_PARTICIONES):
        res, _ = evaluar(*particion_por_sujeto(filas, semilla=semilla))
        for s in res:
            f1s.setdefault(s["caracteristica"], []).append(s["F1"])
    estab = [{"caracteristica": k, "F1_media": round(np.mean(v), 4),
              "F1_desvio": round(np.std(v, ddof=1), 4),
              "F1_min": round(np.min(v), 4), "F1_max": round(np.max(v), 4)}
             for k, v in f1s.items()]
    estab.sort(key=lambda e: -e["F1_media"])
    comun.guardar_csv(estab, os.path.join(config.RES, "roc_umbral_estabilidad.csv"),
                      list(estab[0].keys()))

    print(f"\nF1 en test sobre {N_PARTICIONES} particiones por sujeto")
    print(f"{'caracteristica':18s}{'media':>8s}{'desvio':>8s}{'min':>7s}{'max':>7s}")
    for e in estab:
        print(f"{e['caracteristica']:18s}{e['F1_media']:8.3f}{e['F1_desvio']:8.3f}"
              f"{e['F1_min']:7.3f}{e['F1_max']:7.3f}")
