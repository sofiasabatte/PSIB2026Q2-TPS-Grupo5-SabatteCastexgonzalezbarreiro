# =============================================================================
# RESPONSABLE: Juani
# Hecho (Juani, avance):
#   [x] Correr con los 98 registros y revisar resultados/logs/ si alguno falla.
#   [x] Subir al repo los CSV resumen y las figuras que vayan al informe.
# =============================================================================
# Corre las etapas 2 a 8 para los registros elegidos.
#
# Las etapas 2 a 6 se hacen registro por registro, y se corren N_PROCESOS
# registros en paralelo (config.py). La etapa 7-8 (caracteristicas) se corre
# una sola vez al final con todos los registros juntos.
# Si un registro falla, se anota y se sigue con los demas.
#
# Como correrlo (desde la carpeta principal del repo):
#   python codigo/correr_todo.py            # registros de config.py
#   python codigo/correr_todo.py 001 005    # registros elegidos
#   python codigo/correr_todo.py todos      # los 98 registros anotados
#
# Paralelizacion: se procesan N_PROCESOS registros a la vez (config.py). Por
# defecto usa todos los nucleos de la compu menos uno (deja uno libre para
# que la compu siga usable). Cada registro va en un proceso separado.
# Tiempo aproximado: ~1,5-2,5 min por registro y por proceso. Con 98 registros:
# 4 nucleos -> ~50 min; 8 nucleos -> ~25 min.

import os
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(__file__))
import config
import comun

POR_REGISTRO = ["espectro.py", "preprocesado.py", "deteccion_qrs.py", "serie_rr.py"]


def procesar(reg):
    """Corre las etapas 2 a 6 para un registro. Devuelve (reg, error o None)."""
    os.makedirs(os.path.join(config.RES, "logs"), exist_ok=True)
    log = os.path.join(config.RES, "logs", f"{reg}.txt")
    with open(log, "w") as f:
        for etapa in POR_REGISTRO:
            r = subprocess.run([sys.executable, f"codigo/{etapa}", reg],
                               stdout=f, stderr=subprocess.STDOUT)
            if r.returncode != 0:
                return reg, f"fallo en {etapa} (ver {log})"
    return reg, None


if __name__ == "__main__":
    registros = comun.registros_a_procesar()
    n_proc = config.N_PROCESOS or max(1, (os.cpu_count() or 2) - 1)
    print(f"{len(registros)} registros, {n_proc} en paralelo")
    t0 = time.time()

    errores = {}
    with ProcessPoolExecutor(max_workers=n_proc) as pool:
        futuros = [pool.submit(procesar, reg) for reg in registros]
        for i, fut in enumerate(as_completed(futuros), 1):
            reg, err = fut.result()
            if err:
                errores[reg] = err
            print(f"  [{i}/{len(registros)}] {reg} {'ERROR: ' + err if err else 'ok'}"
                  f"  ({(time.time() - t0) / 60:.1f} min)")

    ok = [r for r in registros if r not in errores]
    print("\n===== caracteristicas.py =====")
    subprocess.run([sys.executable, "codigo/caracteristicas.py", *ok], check=True)

    print(f"\nListo en {(time.time() - t0) / 60:.1f} min. "
          f"{len(ok)} registros ok, {len(errores)} con error.")
    for reg, err in errores.items():
        print(f"  {reg}: {err}")
