# =============================================================================
# ETAPA 3 (extra): espectro ANTES y DESPUES del filtro pasa-banda
#
# Que hace: para un registro (por defecto el 112, el de las diapositivas 16 y 17)
# calcula el espectro de Welch de la misma forma que espectro.py (6 tramos de
# 5 min repartidos en las 24 h, segmentos de 10 s, mediana de los tramos), una
# vez sobre la senal original y otra sobre la senal filtrada con
# comun.filtrar_ecg (Butterworth orden 2, 0,5-40 Hz, filtfilt).
#
# Numeros que imprime y guarda (todos sobre la PSD mediana, 0 a fs/2):
#   - % de la potencia por debajo de 0,5 Hz, entre 0,5 y 40 Hz y arriba de 40 Hz,
#     de la senal original y de la filtrada;
#   - % de la potencia que se conserva al filtrar = potencia filtrada / original.
#
# Entra:  los datos del registro (comun.leer_tramo)
# Sale:   resultados/figuras/espectro_filtro_<reg>.png
#         resultados/espectro_filtro.csv
#
# Como correrlo:
#   python codigo/espectro_filtro.py [registro ...]      (sin argumentos: 112)
# =============================================================================

import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import signal

import config
import comun
from espectro import potencia_banda


def psd_mediana(reg, filtrar):
    fs = config.FS
    n_total = comun.largo_registro(reg)
    largo = 5 * 60 * fs
    inicios = np.linspace(0, n_total - largo, config.N_TRAMOS_ESPECTRO).astype(int)
    psds = []
    for a in inicios:
        x = comun.leer_tramo(reg, a, a + largo)
        x = x - np.mean(x)
        if filtrar:
            x = comun.filtrar_ecg(x, fs)
        f, p = signal.welch(x, fs=fs, window="hann",
                            nperseg=int(config.WELCH_SEG_S * fs))
        psds.append(p)
    return f, np.median(psds, axis=0)


def analizar(reg):
    fs = config.FS
    f, p0 = psd_mediana(reg, filtrar=False)
    _, p1 = psd_mediana(reg, filtrar=True)
    f1, f2 = config.PASABANDA_HZ
    tot0 = potencia_banda(f, p0, 0, fs / 2)
    tot1 = potencia_banda(f, p1, 0, fs / 2)

    def pct(p, a, b, tot):
        return round(float(100 * potencia_banda(f, p, a, b) / tot), 2)

    fila = {
        "registro": reg,
        "antes_<0.5Hz_%": pct(p0, 0, f1, tot0),
        "antes_0.5-40Hz_%": pct(p0, f1, f2, tot0),
        "antes_>40Hz_%": pct(p0, f2, fs / 2, tot0),
        "despues_<0.5Hz_%": pct(p1, 0, f1, tot1),
        "despues_0.5-40Hz_%": pct(p1, f1, f2, tot1),
        "despues_>40Hz_%": pct(p1, f2, fs / 2, tot1),
        "potencia_conservada_%": round(float(100 * tot1 / tot0), 2),
    }

    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.semilogy(f, p0, lw=1.2, label="original")
    ax.semilogy(f, p1, lw=1.2, label="filtrada (0,5-40 Hz)")
    for fr, txt in [(f1, "0,5 Hz"), (f2, "40 Hz"), (50, "50 Hz"), (62.5, "62,5 Hz")]:
        ax.axvline(fr, color="gray", ls="--", lw=0.8)
        ax.text(fr, ax.get_ylim()[1], txt, rotation=90, va="top", fontsize=8)
    ax.set_xlabel("Frecuencia [Hz]")
    ax.set_ylabel("PSD [mV²/Hz]")
    ax.set_title(f"Registro {reg}: espectro antes y después del filtro (Welch, mediana de "
                 f"{config.N_TRAMOS_ESPECTRO} tramos)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    os.makedirs(config.FIG, exist_ok=True)
    ruta = os.path.join(config.FIG, f"espectro_filtro_{reg}.png")
    fig.savefig(ruta, dpi=150)
    plt.close(fig)
    if sys.platform == "darwin":  # en Mac abre la imagen al terminar
        os.system(f'open "{ruta}"')
    print(fila)
    return fila


if __name__ == "__main__":
    regs = sys.argv[1:] or ["112"]
    filas = [analizar(r) for r in regs]
    comun.actualizar_csv(filas, os.path.join(config.RES, "espectro_filtro.csv"),
                         list(filas[0].keys()))
    print("figuras en", config.FIG)
