# =============================================================================
# ETAPAS 3 y 4: resumen de la calidad de la senal (ventanas ruidosas por registro)
#
# Que hace: lee resultados/calidad/<reg>.csv (lo escribe preprocesado.py, una
# fila por ventana de 10 s con bSQI y kSQI) y cuenta, por registro:
#   - ruidosas_bsqi:       ventanas con bSQI < UMBRAL_BSQI (criterio actual)
#   - ruidosas_solo_ksqi:  ventanas con kSQI < UMBRAL_KSQI pero bSQI aceptable
#                          (las que el criterio viejo bsqi_ksqi marcaba de mas)
# Los conteos se recalculan con los umbrales de config.py, asi que no dependen
# del criterio con el que se haya corrido preprocesado.py.
#
# Entra:  resultados/calidad/<reg>.csv  (no se sube a git; se genera con
#         python codigo/preprocesado.py todos)
# Sale:   resultados/resumen_calidad.csv
#         resultados/figuras/resumen_calidad.png
#
# Como correrlo (desde la carpeta principal del repo):
#   python codigo/resumen_calidad.py
# =============================================================================

import csv
import glob
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config

COLUMNAS = ["registro", "ventanas", "ruidosas_bsqi", "pct_ruidosas_bsqi",
            "ruidosas_solo_ksqi", "pct_solo_ksqi"]


def resumen_registro(ruta):
    reg = os.path.splitext(os.path.basename(ruta))[0]
    n = n_b = n_k = 0
    with open(ruta, encoding="utf-8-sig", newline="") as f:
        for fila in csv.DictReader(f):
            bsqi, ksqi = float(fila["bsqi"]), float(fila["ksqi"])
            n += 1
            if bsqi < config.UMBRAL_BSQI:
                n_b += 1
            elif ksqi < config.UMBRAL_KSQI:
                n_k += 1
    return {"registro": reg, "ventanas": n,
            "ruidosas_bsqi": n_b, "pct_ruidosas_bsqi": round(100 * n_b / n, 2),
            "ruidosas_solo_ksqi": n_k, "pct_solo_ksqi": round(100 * n_k / n, 2)}


def figura(filas, ruta):
    orden = sorted(filas, key=lambda f: -f["pct_ruidosas_bsqi"])
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.bar(range(len(orden)), [f["pct_ruidosas_bsqi"] for f in orden], color="C3")
    ax.set_xticks(range(len(orden)))
    ax.set_xticklabels([f["registro"] for f in orden], rotation=90, fontsize=6)
    ax.set_ylabel("Ventanas ruidosas [%]")
    ax.set_xlabel("Registro (ordenados de mas a menos ruidosos)")
    ax.set_title(f"Ventanas de {config.VENTANA_CALIDAD_S} s con bSQI < "
                 f"{config.UMBRAL_BSQI} por registro")
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(ruta, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    rutas = sorted(glob.glob(os.path.join(config.RES, "calidad", "*.csv")))
    rutas = [r for r in rutas
             if os.path.splitext(os.path.basename(r))[0] not in config.REGISTROS_EXCLUIDOS]
    if not rutas:
        raise SystemExit("No hay resultados/calidad/*.csv: correr antes "
                         "python codigo/preprocesado.py todos")
    filas = [resumen_registro(r) for r in rutas]

    total = {"registro": "TOTAL", "ventanas": sum(f["ventanas"] for f in filas),
             "ruidosas_bsqi": sum(f["ruidosas_bsqi"] for f in filas),
             "ruidosas_solo_ksqi": sum(f["ruidosas_solo_ksqi"] for f in filas)}
    total["pct_ruidosas_bsqi"] = round(100 * total["ruidosas_bsqi"] / total["ventanas"], 2)
    total["pct_solo_ksqi"] = round(100 * total["ruidosas_solo_ksqi"] / total["ventanas"], 2)

    ruta = os.path.join(config.RES, "resumen_calidad.csv")
    with open(ruta, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNAS)
        w.writeheader()
        w.writerows(filas + [total])

    os.makedirs(config.FIG, exist_ok=True)
    figura(filas, os.path.join(config.FIG, "resumen_calidad.png"))

    peores = sorted(filas, key=lambda f: -f["pct_ruidosas_bsqi"])[:5]
    print(f"{len(filas)} registros, {total['ventanas']} ventanas")
    print(f"ruidosas (bSQI < {config.UMBRAL_BSQI}): {total['ruidosas_bsqi']} "
          f"({total['pct_ruidosas_bsqi']}%)")
    print(f"solo kSQI (criterio viejo): {total['ruidosas_solo_ksqi']} "
          f"({total['pct_solo_ksqi']}%)")
    print("5 registros mas ruidosos:",
          ", ".join(f"{f['registro']} ({f['pct_ruidosas_bsqi']}%)" for f in peores))
    print("tabla en", ruta)
