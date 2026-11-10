# =============================================================================
# RESPONSABLE: Juani
# ETAPA 5 (figura): Se y VPP de Pan-Tompkins por registro
#
# Que hace: lee resultados/validacion_qrs.csv (lo genera deteccion_qrs.py) y
# grafica la sensibilidad y el VPP de cada registro contra los R del dataset,
# ordenados del peor al mejor. Sirve para el informe: en vez de una tabla de
# 97 filas, se ve de un vistazo que casi todos estan cerca del 100 % y cuales
# son los pocos registros donde el detector falla.
#
# Se usa el subconjunto "todo" (todas las ventanas, tambien las ruidosas),
# porque es donde se ven las diferencias. En "ventanas_limpias" casi todos
# quedan en 100 %.
#
# No necesita la base: solo el CSV que ya esta en el repo.
#
# Entra:  resultados/validacion_qrs.csv
# Sale:   resultados/figuras/validacion_qrs.png
#
# Como correrlo (desde la carpeta principal del repo):
#   python codigo/figura_validacion_qrs.py
# =============================================================================

import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config
import comun

N_PEORES = 8       # registros que se nombran en el titulo y se resaltan


def leer(subconjunto="todo"):
    filas = comun.leer_csv(os.path.join(config.RES, "validacion_qrs.csv"))
    filas = [f for f in filas if f["subconjunto"] == subconjunto
             and f["registro"] not in config.REGISTROS_EXCLUIDOS]
    for f in filas:
        f["Se"] = float(f["Se_%"])
        f["VPP"] = float(f["VPP_%"])
    # orden: el que tenga peor la peor de las dos metricas, primero
    filas.sort(key=lambda f: min(f["Se"], f["VPP"]))
    return filas


def figura(filas, ruta):
    x = np.arange(len(filas))
    se = np.array([f["Se"] for f in filas])
    vpp = np.array([f["VPP"] for f in filas])

    fig, ax = plt.subplots(figsize=(12, 4.5))
    ax.plot(x, se, "o", ms=4, label="Sensibilidad")
    ax.plot(x, vpp, "s", ms=4, mfc="none", label="VPP")
    ax.axhline(99, color="gray", ls="--", lw=0.8)
    ax.text(len(filas) - 1, 99.1, "99 %", ha="right", va="bottom", fontsize=8, color="gray")
    ax.set_xticks(x)
    ax.set_xticklabels([f["registro"] for f in filas], rotation=90, fontsize=6)
    for etiqueta, f in zip(ax.get_xticklabels(), filas):
        if filas.index(f) < N_PEORES:
            etiqueta.set_color("C3")
            etiqueta.set_fontweight("bold")
    ax.set_ylabel("%")
    ax.set_xlabel("Registro (ordenados de peor a mejor)")
    ax.set_title("Pan-Tompkins contra los R del dataset (.qrs), tolerancia 150 ms")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    os.makedirs(os.path.dirname(ruta), exist_ok=True)
    fig.savefig(ruta, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    filas = leer()
    ruta = os.path.join(config.FIG, "validacion_qrs.png")
    figura(filas, ruta)

    se = np.array([f["Se"] for f in filas])
    vpp = np.array([f["VPP"] for f in filas])
    print(f"{len(filas)} registros. Se: mediana {np.median(se):.2f} %, minimo {se.min():.2f} %. "
          f"VPP: mediana {np.median(vpp):.2f} %, minimo {vpp.min():.2f} %.")
    print(f"Registros con Se o VPP < 99 %: {sum(min(a, b) < 99 for a, b in zip(se, vpp))}")
    print("Peores:", ", ".join(f"{f['registro']} (Se {f['Se']:.1f}, VPP {f['VPP']:.1f})"
                              for f in filas[:N_PEORES]))
    print("figura en", ruta)
