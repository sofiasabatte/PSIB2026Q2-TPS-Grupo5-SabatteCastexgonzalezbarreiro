# =============================================================================
# RESPONSABLE: Sofi
# Hecho (Sofi, avance):
#   [x] Revisar tacograma_<reg>.png de varios registros.
#   [x] Comparar cuantos intervalos marca como "ectopico" DENTRO de la FA con
#       QUITAR_ECTOPICOS = True y False (config.py) y como cambian los boxplots.
#   [x] Escribir el parrafo de la etapa 6 para el informe.
# =============================================================================
# ETAPA 6: serie RR, correccion y tacograma
#
# Que hace: a partir de los R de la etapa 5 arma la serie RR y marca cada
# intervalo como valido o no. NO se borra ningun intervalo (se conserva el
# tiempo); solo se marca el motivo:
#   - fuera_rango: RR < RR_MIN_S (falso QRS) o > RR_MAX_S (QRS perdido)
#   - ruido:       el intervalo cae en una ventana ruidosa (etapa 3)
#   - ectopico:    latido ectopico aislado + pausa compensatoria, con el
#                  criterio de Dash et al. (2009): RR[i]/RR[i-1] en el
#                  percentil 1, RR[i+1]/RR[i] en el percentil 99 y
#                  RR[i+1]/RR[i+2] por encima del percentil 25
#   - transicion:  el intervalo cruza un cambio de ritmo
# No se descartan latidos por ser irregulares: la irregularidad es justamente
# lo que caracteriza a la FA.
#
# Entra:  resultados/rpicos/<reg>.npy, resultados/calidad/<reg>.csv, .atr
# Sale:   resultados/rr/<reg>.csv
#         resultados/figuras/tacograma_<reg>.png
#
# Como correrlo:
#   python codigo/serie_rr.py [registros]

import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config
import comun
from deteccion_qrs import mascara_limpia


def ectopicos_dash(rr):
    """Indices i tales que RR[i] (prematuro) y RR[i+1] (pausa) forman un
    ectopico aislado segun Dash et al. 2009."""
    marca = np.zeros(len(rr), dtype=bool)
    if len(rr) < 4:
        return marca
    r1 = rr[1:-2] / rr[:-3]      # RR[i]/RR[i-1]
    r2 = rr[2:-1] / rr[1:-2]     # RR[i+1]/RR[i]
    r3 = rr[2:-1] / rr[3:]       # RR[i+1]/RR[i+2]
    sel = ((r1 <= np.percentile(r1, 1)) & (r2 >= np.percentile(r2, 99))
           & (r3 >= np.percentile(r3, 25)))
    i = np.where(sel)[0] + 1
    marca[i] = True
    marca[i + 1] = True
    return marca


def serie(reg):
    fs = config.FS
    r = np.load(os.path.join(config.RES, "rpicos", f"{reg}.npy"))
    rr = np.diff(r) / fs              # intervalo k = de r[k] a r[k+1]
    t = r[1:] / fs                    # el intervalo se ubica en el latido que lo cierra

    inicios, etiquetas = comun.cambios_de_ritmo(reg)
    et_ini = comun.etiqueta_en(r[:-1], inicios, etiquetas)
    et_fin = comun.etiqueta_en(r[1:], inicios, etiquetas)
    etiqueta = np.where(et_ini == et_fin, et_fin, "transicion")

    motivo = np.full(len(rr), "", dtype=object)
    motivo[(rr < config.RR_MIN_S) | (rr > config.RR_MAX_S)] = "fuera_rango"
    limpio_ini, limpio_fin = mascara_limpia(reg, r[:-1]), mascara_limpia(reg, r[1:])
    if limpio_ini is not None:
        motivo[(motivo == "") & ~(limpio_ini & limpio_fin)] = "ruido"
    if config.QUITAR_ECTOPICOS:
        ok = motivo == ""
        idx_ok = np.where(ok)[0]
        ect = ectopicos_dash(rr[ok])
        motivo[idx_ok[ect]] = "ectopico"
    motivo[(motivo == "") & (etiqueta == "transicion")] = "transicion"

    filas = [{"t_s": round(float(tk), 3), "rr_s": round(float(x), 4),
              "etiqueta": e, "valido": int(m == ""), "motivo": m}
             for tk, x, e, m in zip(t, rr, etiqueta, motivo)]
    comun.guardar_csv(filas, os.path.join(config.RES, "rr", f"{reg}.csv"),
                      list(filas[0].keys()))

    # Resumen
    valido = motivo == ""
    print(f"  {reg}: {len(rr)} intervalos, {valido.sum()} validos "
          f"({100 * valido.mean():.1f}%)")
    for m in ["fuera_rango", "ruido", "ectopico", "transicion"]:
        print(f"      {m:12s} {np.sum(motivo == m)}")
    return t, rr, etiqueta, valido


def figura(reg, t, rr, etiqueta, valido):
    fig, ax = plt.subplots(figsize=(12, 4))
    colores = {"FA": "C1", "noFA": "C0", "otro": "C2"}
    for e, c in colores.items():
        m = (etiqueta == e) & valido
        if m.any():
            ax.plot(t[m] / 3600, rr[m], ".", ms=1, color=c, alpha=0.5, label=e)
    ax.plot(t[~valido] / 3600, rr[~valido], "x", ms=2, color="gray",
            alpha=0.5, label="invalido")
    ax.set_ylim(0, 2.5)
    ax.set_xlabel("Tiempo [h]")
    ax.set_ylabel("RR [s]")
    ax.set_title(f"Registro {reg} – serie RR")
    ax.legend(markerscale=6, fontsize=8, loc="upper right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(config.FIG, f"tacograma_{reg}.png"), dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    os.makedirs(config.FIG, exist_ok=True)
    for reg in comun.registros_a_procesar():
        figura(reg, *serie(reg))
