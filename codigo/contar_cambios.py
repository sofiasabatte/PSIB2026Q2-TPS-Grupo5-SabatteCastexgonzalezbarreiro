"""Cuenta cuantas veces el lado de la marca (arriba/abajo de la mediana)
cambia entre dos latidos consecutivos, en 5 min del medio del registro.

Pocos cambios = la polaridad cambia por bloques (no afecta el RR).
Muchos cambios, cerca de lo esperado al azar = marcas mezcladas latido a latido.

Correr desde la raiz del repo:
    python codigo/contar_cambios.py 021 141 033 041 037
"""
import sys
import numpy as np

import comun
import config

N_MIN = 5
W = int(0.1 * config.FS)

for reg in sys.argv[1:]:
    n = comun.largo_registro(reg)
    L = N_MIN * 60 * config.FS
    a = max(n // 2 - L // 2, 0)
    x = comun.filtrar_ecg(comun.leer_tramo(reg, a, a + L))
    r = comun.pan_tompkins(x)
    arriba = np.array([x[i] > np.median(x[max(i - W, 0):i + W]) for i in r])
    cambios = int(np.sum(arriba[1:] != arriba[:-1]))
    p = arriba.mean()
    azar = 2 * p * (1 - p) * (len(r) - 1)
    print(f"{reg}: {len(r)} latidos | {p*100:.1f}% arriba | "
          f"{cambios} cambios de lado entre consecutivos | "
          f"esperados si estuvieran mezclados al azar: {azar:.0f}")
