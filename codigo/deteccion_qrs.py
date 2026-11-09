# =============================================================================
# RESPONSABLE: Juani
# Hecho (Juani, avance):
#   [x] Revisar validacion_qrs.csv de los 98 registros; mirar qrs_<reg>.png de
#       los que tengan Se o VPP mas bajas.
# =============================================================================
# ETAPA 5: deteccion de QRS (Pan-Tompkins) y validacion
#
# Que hace:
#   - recorre cada registro por tramos, filtra y detecta los R con Pan-Tompkins
#   - compara contra los R del dataset (.qrs) con tolerancia de 150 ms
#     (ANSI/AAMI EC57): sensibilidad (Se) y valor predictivo positivo (VPP)
#   - repite la comparacion solo en ventanas limpias (segun la etapa 3)
#
# OJO: el .qrs del dataset lo genero epltd, que es una implementacion de
# Pan-Tompkins. La referencia NO es independiente: un acuerdo alto indica que
# la implementacion es correcta, no que los latidos esten bien detectados.
#
# Entra:  registros; resultados/calidad/<reg>.csv (etapa 3, opcional)
# Sale:   resultados/rpicos/<reg>.npy       (muestras de los R)
#         resultados/validacion_qrs.csv
#         resultados/figuras/qrs_<reg>.png
#
# Como correrlo:
#   python codigo/deteccion_qrs.py [registros]

import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config
import comun


def detectar(reg):
    n_total = comun.largo_registro(reg)
    r_todos = []
    for a_lec, b_lec, a, b in comun.tramos(n_total):
        x = comun.filtrar_ecg(comun.leer_tramo(reg, a_lec, b_lec))
        r = comun.pan_tompkins(x) + a_lec
        r_todos.append(r[(r >= a) & (r < b)])   # solo la parte util del tramo
    r = np.concatenate(r_todos)
    os.makedirs(os.path.join(config.RES, "rpicos"), exist_ok=True)
    np.save(os.path.join(config.RES, "rpicos", f"{reg}.npy"), r)
    return r


def mascara_limpia(reg, muestras):
    """True para las muestras que caen en ventanas limpias (etapa 3)."""
    ruta = os.path.join(config.RES, "calidad", f"{reg}.csv")
    if not os.path.exists(ruta):
        return None
    filas = comun.leer_csv(ruta)
    ini = np.array([int(f["inicio_muestra"]) for f in filas])
    ruid = np.array([int(f["ruidosa"]) for f in filas])
    idx = np.clip(np.searchsorted(ini, muestras, side="right") - 1, 0, len(ini) - 1)
    return ruid[idx] == 0


def metricas(ref, det):
    tol = int(config.TOL_QRS_S * config.FS)
    # Solo se evalua dentro del tramo que cubre la referencia: el .qrs no
    # anota los primeros segundos ni el final del registro, y lo detectado
    # ahi contaria como falso positivo sin serlo.
    det = det[(det >= ref[0] - tol) & (det <= ref[-1] + tol)]
    vp, fn, fp = comun.emparejar(ref, det, tol)
    return {"VP": vp, "FN": fn, "FP": fp,
            "Se_%": round(100 * vp / (vp + fn), 2) if vp + fn else None,
            "VPP_%": round(100 * vp / (vp + fp), 2) if vp + fp else None}


def figura(reg, r, ref, t0_s=3600):
    fs = config.FS
    a, b = int(t0_s * fs), int((t0_s + 10) * fs)
    x = comun.filtrar_ecg(comun.leer_tramo(reg, a - 5 * fs, b + 5 * fs))[5 * fs:-5 * fs]
    t = np.arange(a, b) / fs
    rr = r[(r >= a) & (r < b)]
    qq = ref[(ref >= a) & (ref < b)]
    fig, ax = plt.subplots(figsize=(11, 3.5))
    ax.plot(t, x, lw=0.8)
    ax.plot(rr / fs, x[rr - a], "rv", label="Pan-Tompkins")
    ax.plot(qq / fs, x[qq - a] * 1.1, "g^", ms=5, label="dataset (.qrs)")
    ax.set_xlabel("Tiempo [s]")
    ax.set_ylabel("mV")
    ax.set_title(f"Registro {reg} – deteccion de QRS")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(config.FIG, f"qrs_{reg}.png"), dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    os.makedirs(config.FIG, exist_ok=True)
    filas = []
    for reg in comun.registros_a_procesar():
        r = detectar(reg)
        ref = comun.leer_anotaciones(reg, "qrs").sample
        fila = {"registro": reg, "subconjunto": "todo", **metricas(ref, r)}
        filas.append(fila)
        print(fila)

        m_ref, m_det = mascara_limpia(reg, ref), mascara_limpia(reg, r)
        if m_ref is not None:
            fila = {"registro": reg, "subconjunto": "ventanas_limpias",
                    **metricas(ref[m_ref], r[m_det])}
            filas.append(fila)
            print(fila)
        figura(reg, r, ref)

    comun.actualizar_csv(filas, os.path.join(config.RES, "validacion_qrs.csv"),
                         list(filas[0].keys()))
