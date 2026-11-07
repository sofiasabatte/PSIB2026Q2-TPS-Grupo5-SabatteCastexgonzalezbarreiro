# =============================================================================
# RESPONSABLE: Pili
# =============================================================================
# ETAPA 4: evaluacion del filtro adaptativo ALE/NLMS sobre ruido REAL
#
# No se agrega ruido artificial: se toman ventanas de 10 s con bSQI < UMBRAL_BSQI
# (dos detectores de QRS que no coinciden = ruido real; ver abajo por que NO se
# usa el kSQI: marca como ruido ECG limpio con ondas T altas), se les aplica
# ale_nlms() con distintos parametros y se mide si mejoran:
#   - bSQI: acuerdo entre Pan-Tompkins y xqrs (criterio del plan: >= 0,8)
#   - Se y VPP de Pan-Tompkins contra los R del dataset (.qrs)
#     OJO: el .qrs lo genero epltd, no es una referencia independiente.
# La ventana se acepta si el bSQI filtrado es >= UMBRAL_BSQI.
#
# Para que el filtro converja, a cada ventana se le dan 20 s de contexto antes
# y 10 s despues; las metricas se miden solo dentro de los 10 s.
#
# Entra:  resultados/calidad/<reg>.csv (etapa 3)
# Sale:   resultados/evaluacion_ale.csv  (una fila por ventana y parametros)
#         El script siempre escribe evaluacion_ale.csv; las dos corridas del informe
#         se guardaron con otro nombre, cada una con las 128 combinaciones
#         (4 delta x 4 L x 8 mu):
#           evaluacion_ale_prueba1.csv       = tanda exploratoria: 021, 077, 113, 131, 137
#           evaluacion_ale_confirmacion.csv  = tanda de confirmacion: 033, 050, 064, 084, 112
#
# Como correrlo (desde la carpeta principal del repo):
#   python codigo/evaluar_ale.py                 # 131, 113, 021, 077, 137; 12 ventanas c/u
#   python codigo/evaluar_ale.py 15 134 137      # N ventanas y registros a elegir

import itertools
import os
import sys

import numpy as np
from wfdb import processing

import config
import comun
from preprocesado import ale_nlms

MU = [round(float(m), 4) for m in np.logspace(np.log10(0.0005), np.log10(0.1), 8)]  # plan del informe
GRILLA = list(itertools.product((5, 10, 20, 40), (32, 64, 128, 256), MU))  # delta, L, mu
CONTEXTO_ANTES_S = 20
CONTEXTO_DESPUES_S = 10


def metricas(x, v0, a, ref):
    """bSQI y (Se, VPP) dentro de la ventana [v0, v0 + 10 s). x arranca en la muestra a."""
    fs = config.FS
    tol = int(config.TOL_QRS_S * fs)
    v1 = v0 + int(config.VENTANA_CALIDAD_S * fs)
    r_pt = comun.pan_tompkins(x) + a
    r_xq = processing.xqrs_detect(x, fs=fs, verbose=False) + a
    p = r_pt[(r_pt >= v0) & (r_pt < v1)]
    q = r_xq[(r_xq >= v0) & (r_xq < v1)]
    vp, fn, fp = comun.emparejar(q, p, tol)
    bsqi = vp / (vp + fn + fp) if (vp + fn + fp) else 0.0
    r = ref[(ref >= v0) & (ref < v1)]
    vp, fn, fp = comun.emparejar(r, p, tol)
    se = vp / (vp + fn) if (vp + fn) else np.nan
    vpp = vp / (vp + fp) if (vp + fp) else np.nan
    return bsqi, se, vpp


def evaluar_registro(reg, n_ventanas, rng):
    fs = config.FS
    calidad = comun.leer_csv(os.path.join(config.RES, "calidad", f"{reg}.csv"))
    ruidosas = [f for f in calidad if float(f["bsqi"]) < config.UMBRAL_BSQI
                and int(f["inicio_muestra"]) >= CONTEXTO_ANTES_S * fs]
    if not ruidosas:
        print(f"  {reg}: sin ventanas ruidosas utilizables")
        return []
    elegidas = rng.choice(len(ruidosas), size=min(n_ventanas, len(ruidosas)), replace=False)
    ref = np.asarray(comun.leer_anotaciones(reg, "qrs").sample)
    n_total = comun.largo_registro(reg)

    filas = []
    for k, idx in enumerate(sorted(elegidas), start=1):
        v0 = int(ruidosas[idx]["inicio_muestra"])
        a = v0 - CONTEXTO_ANTES_S * fs
        fin = min(v0 + (10 + CONTEXTO_DESPUES_S) * fs, n_total)
        x = comun.filtrar_ecg(comun.leer_tramo(reg, a, fin))
        b0, se0, vpp0 = metricas(x, v0, a, ref)
        for delta, L, mu in GRILLA:
            y, _ = ale_nlms(x, delta, L, mu)
            b1, se1, vpp1 = metricas(y, v0, a, ref)
            filas.append({"registro": reg, "inicio_muestra": v0,
                          "delta": delta, "L": L, "mu": mu,
                          "bsqi_antes": round(b0, 3), "bsqi_despues": round(b1, 3),
                          "se_antes": round(se0, 3), "se_despues": round(se1, 3),
                          "vpp_antes": round(vpp0, 3), "vpp_despues": round(vpp1, 3),
                          "aceptada": int(b1 >= config.UMBRAL_BSQI)})
        print(f"  {reg}: ventana {k}/{len(elegidas)}", end="\r")
    print(f"  {reg}: {len(elegidas)} ventanas evaluadas      ")
    return filas


if __name__ == "__main__":
    args = sys.argv[1:]
    n = int(args[0]) if args and args[0].isdigit() else 12
    regs = [a for a in args if not (a.isdigit() and len(a) < 3)] or ["131", "113", "021", "077", "137"]
    rng = np.random.default_rng(0)     # semilla fija: mismas ventanas en cada corrida

    filas = []
    for reg in regs:
        filas += evaluar_registro(reg, n, rng)
    if not filas:
        sys.exit("No hubo ventanas para evaluar (falta resultados/calidad/<reg>.csv?)")

    comun.guardar_csv(filas, os.path.join(config.RES, "evaluacion_ale.csv"), list(filas[0].keys()))

    print(f"\n{'delta':>5s}{'L':>5s}{'mu':>7s} | {'bSQI antes':>10s}{'despues':>9s} | "
          f"{'Se antes':>8s}{'despues':>9s} | {'VPP antes':>9s}{'despues':>9s} | {'aceptadas':>9s}")
    for delta, L, mu in GRILLA:
        f = [r for r in filas if (r["delta"], r["L"], r["mu"]) == (delta, L, mu)]
        med = lambda c: np.nanmedian([r[c] for r in f])
        acept = 100 * np.mean([r["aceptada"] for r in f])
        print(f"{delta:5d}{L:5d}{mu:7.4f} | {med('bsqi_antes'):10.3f}{med('bsqi_despues'):9.3f} | "
              f"{med('se_antes'):8.3f}{med('se_despues'):9.3f} | "
              f"{med('vpp_antes'):9.3f}{med('vpp_despues'):9.3f} | {acept:8.1f}%")
    f0 = [r for r in filas if (r["delta"], r["L"], r["mu"]) == GRILLA[0]]
    ya_ok = 100 * np.mean([r["bsqi_antes"] >= config.UMBRAL_BSQI for r in f0])
    print(f"\n{len(f0)} ventanas con bSQI bajo evaluadas en total.")
    print(f"Sin filtrar, {ya_ok:.1f}% ya llegaba a bSQI >= {config.UMBRAL_BSQI} al medirlas con contexto: "
          f"comparar con 'aceptadas'.")
