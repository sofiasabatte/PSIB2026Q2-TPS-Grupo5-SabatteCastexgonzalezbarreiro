"""Dibuja 10 s de un registro con las marcas de pan_tompkins, en una zona donde
hay marcas de los dos lados (si las hay). Guarda un PNG por registro en la
carpeta temporal del sistema e imprime la ruta.

Correr desde la raiz del repo:
    python codigo/ver_polaridad.py 021 141 033
"""
import os
import sys
import tempfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import comun
import config

N_MIN = 5
W = int(0.1 * config.FS)
FS = config.FS

for reg in sys.argv[1:]:
    n = comun.largo_registro(reg)
    L = N_MIN * 60 * FS
    a0 = max(n // 2 - L // 2, 0)
    x = comun.filtrar_ecg(comun.leer_tramo(reg, a0, a0 + L))
    r = comun.pan_tompkins(x)
    arriba = np.array([x[i] > np.median(x[max(i - W, 0):i + W]) for i in r])
    minoria = ~arriba if arriba.mean() >= 0.5 else arriba
    if minoria.any():
        centro = r[np.where(minoria)[0][len(np.where(minoria)[0]) // 2]]
    else:
        centro = r[len(r) // 2]
    ini = int(np.clip(centro - 5 * FS, 0, len(x) - 10 * FS))
    fin = ini + 10 * FS
    t = np.arange(ini, fin) / FS
    sel = (r >= ini) & (r < fin)
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(t, x[ini:fin], lw=0.8, color="k")
    ax.plot(r[sel & arriba] / FS, x[r[sel & arriba]], "^", color="tab:green",
            label="marca arriba de la mediana")
    ax.plot(r[sel & ~arriba] / FS, x[r[sel & ~arriba]], "v", color="tab:red",
            label="marca abajo de la mediana")
    ax.set_xlabel("Tiempo en el tramo [s]")
    ax.set_ylabel("ECG filtrado [mV]")
    ax.set_title(f"Registro {reg}: {arriba.mean()*100:.1f}% de marcas arriba "
                 f"(5 min del medio)")
    ax.legend(loc="upper right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    ruta = os.path.join(tempfile.gettempdir(), f"polaridad_{reg}.png")
    fig.savefig(ruta, dpi=150)
    plt.close(fig)
    print(f"{reg}: figura en {ruta}")
