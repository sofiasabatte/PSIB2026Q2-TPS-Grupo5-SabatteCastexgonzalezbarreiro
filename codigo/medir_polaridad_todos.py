"""Chequeo del bug de la R en todos los registros anotados.

Por cada registro toma 5 minutos del medio, detecta con pan_tompkins y mide:
  - lado_dominante: % de marcas que caen del lado mas comun (arriba o abajo)
  - RMSSD con la ubicacion actual y con polaridad fija
Escribe resultados/polaridad_r.csv ordenado de mas a menos sospechoso.

Correr desde la raiz del repo:
    python codigo/medir_polaridad_todos.py
"""
import csv
import numpy as np

import comun
import config

N_MIN = 5
W = int(0.1 * config.FS)


def rmssd(r):
    rr = np.diff(r) / config.FS
    rr = rr[(rr > 0.25) & (rr < 2.5)]
    return np.sqrt(np.mean(np.diff(rr) ** 2)) * 1000 if len(rr) > 2 else np.nan


def r_polaridad_fija(x, r_actual):
    segs = []
    for q in r_actual:
        a, b = max(q - W, 0), min(q + W, len(x))
        segs.append((a, x[a:b] - np.median(x[a:b])))
    signo = 1 if np.median([s.max() + s.min() for _, s in segs]) >= 0 else -1
    return np.unique([a + int(np.argmax(signo * s)) for a, s in segs])


filas = []
regs = [r for r in comun.todos_los_registros() if r not in config.REGISTROS_EXCLUIDOS]
for k, reg in enumerate(regs, 1):
    try:
        n = comun.largo_registro(reg)
        L = N_MIN * 60 * config.FS
        a = max(n // 2 - L // 2, 0)
        x = comun.filtrar_ecg(comun.leer_tramo(reg, a, a + L))
        r = comun.pan_tompkins(x)
        arriba = np.array([x[i] > np.median(x[max(i - W, 0):i + W]) for i in r])
        dom = max(arriba.mean(), 1 - arriba.mean()) * 100
        r2 = r_polaridad_fija(x, r)
        ra, rb = rmssd(r), rmssd(r2)
        filas.append({"registro": reg, "latidos": len(r),
                      "lado_dominante_pct": round(dom, 1),
                      "rmssd_actual_ms": round(ra, 1),
                      "rmssd_polaridad_fija_ms": round(rb, 1),
                      "dif_ms": round(ra - rb, 1)})
        print(f"[{k}/{len(regs)}] {reg}: {dom:.1f}% mismo lado, dif RMSSD {ra - rb:+.1f} ms")
    except Exception as e:
        print(f"[{k}/{len(regs)}] {reg}: error {e}")

filas.sort(key=lambda f: f["lado_dominante_pct"])
ruta = "resultados/polaridad_r.csv"
with open(ruta, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(filas[0].keys()))
    w.writeheader()
    w.writerows(filas)

print("\nLos 10 con menos marcas del mismo lado:")
for f in filas[:10]:
    print(f"  {f['registro']}: {f['lado_dominante_pct']}% | dif RMSSD {f['dif_ms']:+} ms")
print(f"\nTabla completa en {ruta}")
