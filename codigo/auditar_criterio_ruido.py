# =============================================================================
# ETAPA 3: auditoria del criterio de ruido (bSQI contra kSQI)
#
# Que hace: clasifica cada ventana de 10 s en cuatro grupos segun quien la marca
# como ruidosa (con los umbrales de config.py):
#   limpias:    bSQI >= UMBRAL_BSQI y kSQI >= UMBRAL_KSQI
#   solo_ksqi:  la marca solo el kSQI (bSQI aceptable)
#   solo_bsqi:  la marca solo el bSQI (kSQI aceptable)
#   ambos:      la marcan los dos
# y mide, en cada grupo, que tan bien detecta Pan-Tompkins contra las
# anotaciones .qrs de la base:
#   Se:  sensibilidad = VP / (VP + FN). De los latidos reales, cuantos encuentra.
#   VPP: VP / (VP + FP). De los que marca, cuantos son reales.
# Se informan dos versiones:
#   Se, VPP:                 con los VP, FN y FP sumados sobre todas las ventanas del grupo
#   Se_mediana, VPP_mediana: mediana de la Se y del VPP de cada ventana
# "pct_bueno" es el porcentaje de ventanas del grupo donde Se y VPP dieron
# las dos 0,95 o mas (no cuentan las ventanas sin latidos de referencia ni
# detectados, porque no hay nada que juzgar).
#
# Conclusion que sostiene la decision de config.CRITERIO_RUIDO = "bsqi": si en
# las ventanas "solo_ksqi" el detector anda igual que en las limpias, el kSQI
# las marcaba sin que hubiera ruido real (eran ondas T altas).
#
# Entra:  resultados/calidad/<reg>.csv  (etapa 3, python codigo/preprocesado.py todos)
#         resultados/rpicos/<reg>.npy   (etapa 5, python codigo/deteccion_qrs.py)
#         anotaciones .qrs de la base
# Sale:   resultados/auditoria_ruido.csv
#         resultados/figuras/auditoria_ruido.png
#
# Como correrlo (desde la carpeta principal del repo):
#   python codigo/auditar_criterio_ruido.py
# =============================================================================

import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config
import comun

GRUPOS = ["limpias", "solo_ksqi", "solo_bsqi", "ambos"]
UMBRAL_BUENO = 0.95
COLUMNAS = ["grupo", "ventanas", "VP", "FN", "FP", "Se", "VPP",
            "Se_mediana", "VPP_mediana", "pct_bueno"]


def grupo_de(bsqi, ksqi):
    mal_b = bsqi < config.UMBRAL_BSQI
    mal_k = ksqi < config.UMBRAL_KSQI
    if mal_b and mal_k:
        return "ambos"
    if mal_b:
        return "solo_bsqi"
    if mal_k:
        return "solo_ksqi"
    return "limpias"


def auditar_registro(reg, acum):
    """Suma los VP, FN y FP de cada ventana del registro en acum[grupo]."""
    ruta_c = os.path.join(config.RES, "calidad", f"{reg}.csv")
    ruta_r = os.path.join(config.RES, "rpicos", f"{reg}.npy")
    if not (os.path.exists(ruta_c) and os.path.exists(ruta_r)):
        return False
    filas = comun.leer_csv(ruta_c)
    det = np.load(ruta_r)
    ref = np.asarray(comun.leer_anotaciones(reg, "qrs").sample)
    tol = int(config.TOL_QRS_S * config.FS)
    # Igual que en deteccion_qrs.py: solo se evalua donde hay referencia.
    ini_ref, fin_ref = ref[0] - tol, ref[-1] + tol

    for f in filas:
        v0, v1 = int(f["inicio_muestra"]), int(f["fin_muestra"])
        if v0 < ini_ref or v1 > fin_ref:
            continue
        g = grupo_de(float(f["bsqi"]), float(f["ksqi"]))
        r = ref[np.searchsorted(ref, v0):np.searchsorted(ref, v1)]
        d = det[np.searchsorted(det, v0):np.searchsorted(det, v1)]
        vp, fn, fp = comun.emparejar(r, d, tol)
        a = acum[g]
        a["ventanas"] += 1
        a["VP"] += vp
        a["FN"] += fn
        a["FP"] += fp
        if vp + fn:
            a["se_v"].append(vp / (vp + fn))
        if vp + fp:
            a["vpp_v"].append(vp / (vp + fp))
        if vp + fn + fp > 0:                      # hay algo que juzgar
            a["juzgadas"] += 1
            se = vp / (vp + fn) if vp + fn else 0.0
            vpp = vp / (vp + fp) if vp + fp else 0.0
            if se >= UMBRAL_BUENO and vpp >= UMBRAL_BUENO:
                a["buenas"] += 1
    return True


def figura(filas, ruta):
    x = np.arange(len(filas))
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(x - 0.2, [f["Se"] for f in filas], 0.4, label="Sensibilidad")
    ax.bar(x + 0.2, [f["VPP"] for f in filas], 0.4, label="VPP")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{f['grupo']}\n({f['ventanas']})" for f in filas])
    ax.set_ylim(0, 1.05)
    ax.set_title("Pan-Tompkins contra .qrs segun quien marca la ventana")
    ax.legend(loc="lower left")
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(ruta, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    acum = {g: {"ventanas": 0, "VP": 0, "FN": 0, "FP": 0, "juzgadas": 0, "buenas": 0,
                "se_v": [], "vpp_v": []}
            for g in GRUPOS}
    regs = [r for r in comun.todos_los_registros() if r not in config.REGISTROS_EXCLUIDOS]
    hechos = 0
    for k, reg in enumerate(regs, 1):
        if auditar_registro(reg, acum):
            hechos += 1
        print(f"  {k}/{len(regs)} registros", end="\r")
    print()
    if not hechos:
        raise SystemExit("Faltan resultados/calidad/*.csv y resultados/rpicos/*.npy: "
                         "correr antes preprocesado.py y deteccion_qrs.py")

    filas = []
    for g in GRUPOS:
        a = acum[g]
        vp, fn, fp = a["VP"], a["FN"], a["FP"]
        filas.append({
            "grupo": g, "ventanas": a["ventanas"], "VP": vp, "FN": fn, "FP": fp,
            "Se": round(vp / (vp + fn), 3) if vp + fn else float("nan"),
            "VPP": round(vp / (vp + fp), 3) if vp + fp else float("nan"),
            "Se_mediana": round(float(np.median(a["se_v"])), 3) if a["se_v"] else float("nan"),
            "VPP_mediana": round(float(np.median(a["vpp_v"])), 3) if a["vpp_v"] else float("nan"),
            "pct_bueno": round(100 * a["buenas"] / a["juzgadas"], 1) if a["juzgadas"] else float("nan"),
        })
    comun.guardar_csv(filas, os.path.join(config.RES, "auditoria_ruido.csv"), COLUMNAS)
    os.makedirs(config.FIG, exist_ok=True)
    figura(filas, os.path.join(config.FIG, "auditoria_ruido.png"))

    print(f"{hechos} registros")
    for f in filas:
        print(f"{f['grupo']:10s} {f['ventanas']:8d} ventanas | Se {f['Se']} | "
              f"VPP {f['VPP']} | mediana Se {f['Se_mediana']} VPP {f['VPP_mediana']} | "
              f"{f['pct_bueno']}% buenas")
    print("tabla en", os.path.join(config.RES, "auditoria_ruido.csv"))
