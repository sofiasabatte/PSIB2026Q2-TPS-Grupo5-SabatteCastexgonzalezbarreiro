# =============================================================================
# RESPONSABLE: Pili
# Hecho (Pili, avance):
#   [x] Correr en todos los registros y revisar espectro_resumen.csv: notch en
#       50 Hz (pico_50Hz_x < 1) y pozo en 75 Hz en todos.
# =============================================================================
# ETAPA 2: inspeccion espectral
#
# Que hace: estima la PSD con Welch en varios tramos de 5 min repartidos en
# las 24 h de cada registro y la grafica.
# Por que: para decidir que filtrar. Hay que verificar:
#   - si la senal ya viene limitada en banda (grabada a 125 Hz -> no deberia
#     haber energia por encima de ~62,5 Hz; ahi estaria el "pozo")
#   - si hay pico de red en 50 o 60 Hz (en Japon se usan las dos)
#
# Entra:  registros de config.REGISTROS (o los pasados por linea de comandos)
# Sale:   resultados/figuras/espectro_<reg>.png
#         resultados/espectro_resumen.csv
#
# Como correrlo (desde la carpeta principal del repo):
#   python codigo/espectro.py            # registros de config.py
#   python codigo/espectro.py 001 005    # registros elegidos

import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import signal

import config
import comun


def potencia_banda(f, p, f1, f2):
    m = (f >= f1) & (f < f2)
    return np.trapezoid(p[m], f[m]) if hasattr(np, "trapezoid") else np.trapz(p[m], f[m])


def analizar(reg):
    fs = config.FS
    n_total = comun.largo_registro(reg)
    largo = 5 * 60 * fs
    inicios = np.linspace(0, n_total - largo, config.N_TRAMOS_ESPECTRO).astype(int)

    psds = []
    for a in inicios:
        x = comun.leer_tramo(reg, a, a + largo)
        x = x - np.mean(x)
        f, p = signal.welch(x, fs=fs, window="hann",
                            nperseg=int(config.WELCH_SEG_S * fs))  # 50% solape por defecto
        psds.append(p)
    psds = np.array(psds)

    fig, ax = plt.subplots(figsize=(10, 4.5))
    for a, p in zip(inicios, psds):
        ax.semilogy(f, p, lw=0.7, alpha=0.6, label=f"{a / fs / 3600:.1f} h")
    ax.semilogy(f, np.median(psds, axis=0), "k", lw=1.5, label="mediana")
    for fr, txt in [(50, "50 Hz"), (60, "60 Hz"), (62.5, "125/2 Hz")]:
        ax.axvline(fr, color="r" if fr < 62 else "gray", ls="--", lw=0.8)
        ax.text(fr, ax.get_ylim()[1], txt, rotation=90, va="top", fontsize=8)
    ax.set_xlabel("Frecuencia [Hz]")
    ax.set_ylabel("PSD [mV²/Hz]")
    ax.set_title(f"Registro {reg} – Welch (Hann, {config.WELCH_SEG_S} s, 50% solape)")
    ax.legend(fontsize=7, ncol=4)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    os.makedirs(config.FIG, exist_ok=True)
    fig.savefig(os.path.join(config.FIG, f"espectro_{reg}.png"), dpi=150)
    plt.close(fig)

    # Numeros para el informe
    pm = np.median(psds, axis=0)
    total = potencia_banda(f, pm, 0, fs / 2)
    fila = {
        "registro": reg,
        "pot_0.5_40Hz_%": round(100 * potencia_banda(f, pm, 0.5, 40) / total, 2),
        "pot_>62.5Hz_%": round(100 * potencia_banda(f, pm, 62.5, fs / 2) / total, 4),
        # pico de red: PSD en 50/60 Hz relativa a la mediana de 45-55 / 55-65 Hz
        "pico_50Hz_x": round(pm[np.argmin(abs(f - 50))] / np.median(pm[(f > 45) & (f < 55)]), 2),
        "pico_60Hz_x": round(pm[np.argmin(abs(f - 60))] / np.median(pm[(f > 55) & (f < 62)]), 2),
    }
    # Imagen especular: grabado a 125 Hz y remuestreado a 200 Hz, lo que hay por
    # encima de 62,5 Hz es el espejo de lo que hay por debajo (f -> 125 - f).
    # pozo_75Hz_x < 1: hay un pozo en 75 Hz (espejo del notch de 50 Hz).
    # corr_espejo alta: el espectro de 62,5 a 100 Hz copia el de 25 a 62,5 Hz.
    m = (f > 62.5) & (f < 100)
    espejo = np.interp(125 - f[m], f, pm)
    fila["pozo_75Hz_x"] = round(pm[np.argmin(abs(f - 75))] / np.median(pm[(f > 70) & (f < 80)]), 2)
    fila["corr_espejo"] = round(np.corrcoef(np.log(pm[m]), np.log(espejo))[0, 1], 3)
    print(fila)
    return fila


if __name__ == "__main__":
    filas = [analizar(reg) for reg in comun.registros_a_procesar()]
    comun.actualizar_csv(filas, os.path.join(config.RES, "espectro_resumen.csv"),
                         list(filas[0].keys()))
    print("figuras en", config.FIG)
