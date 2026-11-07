# =============================================================================
# ETAPAS 6 y 7: comparaciones de decisiones de diseno (reproducen los CSV)
#
# Dos comparaciones, con los mismos registros y la misma etapa 8:
#
#   ectopicos: QUITAR_ECTOPICOS = True contra False (config.py).
#              Mediana de cada caracteristica por clase (sobre todas las
#              ventanas) y cambio porcentual al NO quitar los ectopicos.
#   ventanas:  VENTANA_LATIDOS = 60 contra 128.
#              AUC por sujeto (mediana por sujeto y clase, FA = positivo;
#              se informa max(AUC, 1 - AUC)).
#
# Entra:  resultados/rpicos/<reg>.npy (etapa 5), calidad y anotaciones.
# Sale:   resultados/comparacion_ectopicos.csv
#         resultados/comparacion_ventanas.csv
#
# OJO: "ectopicos" vuelve a escribir resultados/rr/<reg>.csv; termina con el
# valor de config.QUITAR_ECTOPICOS para que esa carpeta quede como estaba.
# No toca resultados/caracteristicas.csv.
#
# Como correrlo (desde la carpeta principal del repo):
#   python codigo/comparar_config.py ectopicos
#   python codigo/comparar_config.py ventanas
# Usa varios procesos (uno por registro) y tarda varios minutos.
# =============================================================================

import contextlib
import io
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from scipy import stats

import config
import comun
from caracteristicas import NOMBRES, ventanas_registro
from estadistica import por_sujeto
from serie_rr import serie


def registros():
    regs = [r for r in comun.todos_los_registros() if r not in config.REGISTROS_EXCLUIDOS]
    return [r for r in regs
            if os.path.exists(os.path.join(config.RES, "rpicos", f"{r}.npy"))]


def procesar(reg, quitar, n_latidos, rehacer_serie):
    """Corre la serie RR (si hace falta) y las ventanas de un registro con la
    configuracion pedida. Devuelve las ventanas como filas de texto."""
    config.QUITAR_ECTOPICOS = quitar
    config.VENTANA_LATIDOS = n_latidos
    with contextlib.redirect_stdout(io.StringIO()):
        if rehacer_serie:
            serie(reg)
        filas = ventanas_registro(reg, comun.subject_ids())
    return [{"sujeto": f["sujeto"], "etiqueta": f["etiqueta"],
             **{n: str(f[n]) for n in NOMBRES}} for f in filas]


def correr(regs, quitar, n_latidos, rehacer_serie):
    filas = []
    with ProcessPoolExecutor() as pool:
        futuros = [pool.submit(procesar, r, quitar, n_latidos, rehacer_serie)
                   for r in regs]
        for k, fu in enumerate(futuros, 1):
            filas += fu.result()
            print(f"  {k}/{len(regs)} registros", end="\r")
    print()
    return filas


def medianas_por_clase(filas):
    et = np.array([f["etiqueta"] for f in filas])
    sal = {}
    for n in NOMBRES:
        v = np.array([float(f[n]) for f in filas])
        sal[n] = {e: float(np.nanmedian(v[et == e])) for e in ("noFA", "FA")}
    return sal


def auc_por_sujeto(filas):
    agregado = por_sujeto(filas)
    sujetos = {s for (s, _) in agregado}
    sal = {}
    for n in NOMBRES:
        fa = [agregado[(s, "FA")][n] for s in sujetos if (s, "FA") in agregado]
        nofa = [agregado[(s, "noFA")][n] for s in sujetos if (s, "noFA") in agregado]
        fa, nofa = np.array(fa), np.array(nofa)
        fa, nofa = fa[~np.isnan(fa)], nofa[~np.isnan(nofa)]
        auc = stats.mannwhitneyu(fa, nofa).statistic / (len(fa) * len(nofa))
        sal[n] = max(auc, 1 - auc)
    return sal


def comparar_ectopicos(regs):
    resultados = {}
    orden = [not config.QUITAR_ECTOPICOS, config.QUITAR_ECTOPICOS]   # termina con el valor de config
    for quitar in orden:
        print(f"QUITAR_ECTOPICOS = {quitar}")
        filas = correr(regs, quitar, config.VENTANA_LATIDOS, True)
        resultados[quitar] = medianas_por_clase(filas)
        n_fa = sum(f["etiqueta"] == "FA" for f in filas)
        print(f"  {len(filas)} ventanas ({n_fa} FA)")
    tabla = []
    for n in NOMBRES:
        t, f = resultados[True][n], resultados[False][n]
        cambio = lambda a, b: round(100 * (b - a) / a, 1) if a else float("nan")
        tabla.append({"caracteristica": n,
                      "noFA_QUITAR_True": round(t["noFA"], 4), "FA_QUITAR_True": round(t["FA"], 4),
                      "noFA_QUITAR_False": round(f["noFA"], 4), "FA_QUITAR_False": round(f["FA"], 4),
                      "cambio_noFA_%": cambio(t["noFA"], f["noFA"]),
                      "cambio_FA_%": cambio(t["FA"], f["FA"])})
    ruta = os.path.join(config.RES, "comparacion_ectopicos.csv")
    comun.guardar_csv(tabla, ruta, list(tabla[0].keys()))
    print("guardado en", ruta)


def comparar_ventanas(regs):
    auc = {}
    for n_lat in (60, 128):
        print(f"VENTANA_LATIDOS = {n_lat}")
        filas = correr(regs, config.QUITAR_ECTOPICOS, n_lat, False)
        auc[n_lat] = auc_por_sujeto(filas)
        n_fa = sum(f["etiqueta"] == "FA" for f in filas)
        print(f"  {len(filas)} ventanas ({n_fa} FA, {len(filas) - n_fa} noFA)")
    tabla = [{"caracteristica": n,
              "AUC_60_latidos": round(auc[60][n], 3),
              "AUC_128_latidos": round(auc[128][n], 3),
              "diferencia_128_menos_60": round(auc[128][n] - auc[60][n], 3)}
             for n in NOMBRES]
    ruta = os.path.join(config.RES, "comparacion_ventanas.csv")
    comun.guardar_csv(tabla, ruta, list(tabla[0].keys()))
    print("guardado en", ruta)


if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in ("ectopicos", "ventanas"):
        sys.exit("Uso: python codigo/comparar_config.py ectopicos | ventanas")
    regs = registros()
    print(f"{len(regs)} registros")
    if sys.argv[1] == "ectopicos":
        comparar_ectopicos(regs)
    else:
        comparar_ventanas(regs)
