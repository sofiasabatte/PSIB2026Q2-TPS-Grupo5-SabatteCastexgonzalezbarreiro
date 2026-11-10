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
#      reporta media, desvio, minimo y maximo del F1 en test, y media y desvio
#      de Se y Esp. Con 93 sujetos el resultado de UNA particion depende mucho
#      de que sujetos caen en test. El F1 depende ademas de cuanta FA cae en
#      test (prevalencia); Se y Esp no, por eso se informan aparte.
#   5b. Regla de Dash AJUSTADA a 60 latidos: misma forma de la regla, pero los
#      4 umbrales (nrmssd, shannon, tpr minimo y maximo) se eligen en TRAIN por
#      busqueda en grilla maximizando el indice de Youden, y se evaluan en TEST.
#      Responde a que los umbrales publicados son para ventanas de 128 latidos.
#   6. Falsos positivos por registro: en cada particion, las ventanas no-FA del
#      test que la regla de Dash marca como FA, sumadas por registro. Sirve para
#      ver si los errores se reparten o se concentran en pocos registros.
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
#         resultados/figuras/roc_estabilidad.png   (F1 en test, una caja por caracteristica)
#         resultados/roc_falsos_positivos.csv      (regla de Dash, por registro)
#         resultados/roc_dash_ajustada.csv         (umbrales elegidos en cada particion)
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
REGLA_DASH_AJ = "regla Dash ajustada"
# Grilla para ajustar la regla de Dash en train (incluye los valores publicados).
# TPR maximo = 1,0 equivale a no poner limite superior.
GRILLA_DASH = {
    "nrmssd": np.round(np.arange(0.05, 0.205, 0.01), 3),
    "shannon": np.round(np.arange(0.60, 0.905, 0.02), 3),
    "tpr_min": np.round(np.arange(0.36, 0.605, 0.02), 3),
    "tpr_max": np.array([0.70, 0.77, 0.85, 1.0]),
}


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


def falsos_positivos_dash(test, acum):
    """Suma en acum[registro] las ventanas no-FA del test y cuantas de ellas
    la regla de Dash marca como FA."""
    pred, y = regla_dash(test)
    for f, p, es_fa in zip(test, pred, y):
        if es_fa:
            continue
        d = acum.setdefault(f["registro"], {"sujeto": f["sujeto"], "noFA": 0, "FP": 0, "particiones": set()})
        d["noFA"] += 1
        d["FP"] += int(p)


def _arrays_dash(filas):
    nr = np.array([float(f["nrmssd"]) for f in filas])
    tp = np.array([float(f["tpr"]) for f in filas])
    sh = np.array([float(f["shannon"]) for f in filas])
    y = np.array([f["etiqueta"] == "FA" for f in filas])
    return nr, tp, sh, y


def ajustar_dash(train):
    """Umbrales de la regla de Dash que maximizan Youden en train.
    Devuelve (nrmssd, shannon, tpr_min, tpr_max)."""
    nr, tp, sh, y = _arrays_dash(train)
    mejor_j, mejor = -1.0, None
    for a in GRILLA_DASH["nrmssd"]:
        ma = nr > a
        for b in GRILLA_DASH["shannon"]:
            mb = ma & (sh > b)
            for c in GRILLA_DASH["tpr_min"]:
                mc = mb & (tp >= c)
                for d in GRILLA_DASH["tpr_max"]:
                    p = mc & (tp <= d)
                    j = p[y].mean() + (~p[~y]).mean() - 1
                    if j > mejor_j:
                        mejor_j, mejor = j, (float(a), float(b), float(c), float(d))
    return mejor


def aplicar_dash(filas, umbrales):
    a, b, c, d = umbrales
    nr, tp, sh, y = _arrays_dash(filas)
    return (nr > a) & (sh > b) & (tp >= c) & (tp <= d), y


def texto_dash(umbrales):
    a, b, c, d = umbrales
    tmax = "" if d >= 1 else f" y <={d:g}"
    return f"nrmssd>{a:g}, shannon>{b:g}, tpr>={c:g}{tmax}"


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
    umb = ajustar_dash(train)
    pred, y = aplicar_dash(test, umb)
    salida.append({"caracteristica": REGLA_DASH_AJ, "AUC_train": np.nan,
                   "umbral": np.nan, "FA_si": texto_dash(umb),
                   **metricas(pred, y), "_umbrales": umb})
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
    comun.guardar_csv([{k: redondear(v) for k, v in s.items() if not k.startswith("_")}
                       for s in salida],
                      os.path.join(config.RES, "roc_umbral.csv"), list(salida[0].keys()))

    fig, ax = plt.subplots(figsize=(6, 6))
    for s in salida:
        if s["caracteristica"] not in curvas:
            continue
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
    plt.close(fig)

    print(f"\nParticion {SEMILLA} (Se, Esp, VPP y F1 en TEST; AUC en TRAIN)")
    print(f"{'caracteristica':18s}{'AUC':>7s}{'umbral':>9s}{'Se':>7s}{'Esp':>7s}{'VPP':>7s}{'F1':>7s}")
    for s in sorted(salida, key=lambda s: -s["F1"]):
        u = ("fijo" if s["caracteristica"] == REGLA_DASH else
             "ajust." if s["caracteristica"] == REGLA_DASH_AJ else f"{s['FA_si']}{s['umbral']:.3g}")
        a = "" if np.isnan(s["AUC_train"]) else f"{s['AUC_train']:.3f}"
        print(f"{s['caracteristica']:18s}{a:>7s}{u:>9s}{s['Se']:7.3f}{s['Esp']:7.3f}"
              f"{s['VPP']:7.3f}{s['F1']:7.3f}")

    # --- 2. estabilidad: N_PARTICIONES particiones ---------------------------
    por_metrica = {"F1": {}, "Se": {}, "Esp": {}}
    fp_reg = {}
    umbrales_aj = []
    for semilla in range(N_PARTICIONES):
        train_s, test_s = particion_por_sujeto(filas, semilla=semilla)
        res, _ = evaluar(train_s, test_s)
        falsos_positivos_dash(test_s, fp_reg)
        for f in test_s:
            if f["registro"] in fp_reg:
                fp_reg[f["registro"]]["particiones"].add(semilla)
        for s in res:
            if s["caracteristica"] == REGLA_DASH_AJ:
                a, b, c, d_ = s["_umbrales"]
                umbrales_aj.append({"particion": semilla, "nrmssd_mayor_a": a,
                                    "shannon_mayor_a": b, "tpr_min": c,
                                    "tpr_max": d_ if d_ < 1 else "sin limite",
                                    "F1_test": round(s["F1"], 4), "Se_test": round(s["Se"], 4),
                                    "Esp_test": round(s["Esp"], 4)})
            for m, d in por_metrica.items():
                d.setdefault(s["caracteristica"], []).append(s[m])
    f1s = por_metrica["F1"]
    estab = []
    for k, v in f1s.items():
        se, esp = por_metrica["Se"][k], por_metrica["Esp"][k]
        estab.append({"caracteristica": k, "F1_media": round(np.mean(v), 4),
                      "F1_desvio": round(np.std(v, ddof=1), 4),
                      "F1_min": round(np.min(v), 4), "F1_max": round(np.max(v), 4),
                      "Se_media": round(np.mean(se), 4), "Se_desvio": round(np.std(se, ddof=1), 4),
                      "Esp_media": round(np.mean(esp), 4), "Esp_desvio": round(np.std(esp, ddof=1), 4)})
    estab.sort(key=lambda e: -e["F1_media"])
    comun.guardar_csv(estab, os.path.join(config.RES, "roc_umbral_estabilidad.csv"),
                      list(estab[0].keys()))

    print(f"\nTest sobre {N_PARTICIONES} particiones por sujeto (media +- desvio)")
    print(f"{'caracteristica':18s}{'F1':>14s}{'Se':>14s}{'Esp':>14s}")
    for e in estab:
        print(f"{e['caracteristica']:18s}"
              f"{e['F1_media']:8.3f} +-{e['F1_desvio']:.3f}"
              f"{e['Se_media']:8.3f} +-{e['Se_desvio']:.3f}"
              f"{e['Esp_media']:8.3f} +-{e['Esp_desvio']:.3f}")

    comun.guardar_csv(umbrales_aj, os.path.join(config.RES, "roc_dash_ajustada.csv"),
                      list(umbrales_aj[0].keys()))
    print(f"\nRegla de Dash ajustada en train (umbrales por particion):")
    for u in umbrales_aj:
        print(f"  {u['particion']}: nrmssd>{u['nrmssd_mayor_a']:g} shannon>{u['shannon_mayor_a']:g} "
              f"tpr>={u['tpr_min']:g} tpr<={u['tpr_max']}  F1 test {u['F1_test']:.3f}")

    # Figura: una caja por caracteristica con el F1 de las N particiones,
    # ordenadas por mediana. La regla de Dash en otro color.
    orden = sorted(f1s, key=lambda k: np.median(f1s[k]))
    fig, ax = plt.subplots(figsize=(8, 5))
    cajas = ax.boxplot([f1s[k] for k in orden], orientation="horizontal", patch_artist=True,
                       tick_labels=orden, medianprops={"color": "k"})
    for k, caja in zip(orden, cajas["boxes"]):
        caja.set_facecolor({REGLA_DASH: "C3", REGLA_DASH_AJ: "C1"}.get(k, "C0"))
        caja.set_alpha(0.6)
    for i, k in enumerate(orden, start=1):
        ax.plot(f1s[k], np.full(len(f1s[k]), i), "k.", ms=3)
    ax.set_xlabel("F1 en test")
    ax.set_title(f"F1 en test en {N_PARTICIONES} particiones por sujeto")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(config.FIG, "roc_estabilidad.png"), dpi=150)
    plt.close(fig)

    # --- 3. falsos positivos de la regla de Dash por registro ---------------
    # Cada registro cae en test en algunas particiones; se suman todas.
    total_fp = sum(d["FP"] for d in fp_reg.values())
    fp_filas = [{"registro": r, "sujeto": d["sujeto"],
                 "particiones_en_test": len(d["particiones"]),
                 "ventanas_noFA_test": d["noFA"], "FP": d["FP"],
                 "FP_%_del_registro": round(100 * d["FP"] / d["noFA"], 2),
                 "FP_%_del_total": round(100 * d["FP"] / total_fp, 2) if total_fp else 0.0}
                for r, d in fp_reg.items()]
    fp_filas.sort(key=lambda d: -d["FP"])
    comun.guardar_csv(fp_filas, os.path.join(config.RES, "roc_falsos_positivos.csv"),
                      list(fp_filas[0].keys()))

    acum = np.cumsum([d["FP"] for d in fp_filas]) / total_fp if total_fp else []
    n80 = int(np.searchsorted(acum, 0.8) + 1) if total_fp else 0
    print(f"\nFalsos positivos de la regla de Dash en test ({N_PARTICIONES} particiones): {total_fp}")
    print(f"El 80% de los FP sale de {n80} de {len(fp_filas)} registros.")
    print(f"{'registro':>9s}{'FP':>8s}{'% del reg':>11s}{'% del total':>13s}")
    for d in fp_filas[:10]:
        print(f"{d['registro']:>9s}{d['FP']:8d}{d['FP_%_del_registro']:11.2f}{d['FP_%_del_total']:13.2f}")
