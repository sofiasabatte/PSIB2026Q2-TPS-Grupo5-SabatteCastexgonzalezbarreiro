# Detección de fibrilación auricular con HRV (SHDB-AF)

TP de Procesamiento de Señales e Imágenes Biomédicas (ITBA), Grupo 5.

> **Versión provisoria.** Falta revisar con el equipo los nombres de los integrantes, la bibliografía y los resultados finales.

## De qué trata

La fibrilación auricular (FA) hace que el ritmo del corazón sea irregular. Este trabajo la detecta **mirando solo el ritmo**: calcula 10 medidas sobre los tiempos entre latidos (RR) en ventanas de 60 latidos y mide qué tan bien separa cada una la FA del ritmo normal. No usa machine learning.

Datos: base SHDB-AF de PhysioNet (Tsutsui et al., 2025), 128 Holters de 24 h, de los cuales 98 tienen el ritmo anotado, de 93 sujetos distintos (el 005 y el 020 son el mismo archivo y se excluye el 020).

## Cómo prepararlo

1. Python 3.12 y las librerías de `requirements.txt`:
   ```
   pip install -r requirements.txt
   ```
2. Bajar la base SHDB-AF de PhysioNet y ponerla en `data/shdb-af/` (con los `.hea`, `.dat`, `.atr` y `.qrs`). La carpeta `data/` no se sube a git. Si un registro no está ahí, se lee directo de PhysioNet.
3. Correr siempre **desde la carpeta principal del repo**.

Todos los parámetros están en `codigo/config.py` (filtro, umbrales de ruido, límites del RR, tamaño de ventana, registros excluidos). Se cambian ahí y no en los scripts.

## Cómo correrlo, etapa por etapa

| Etapa | Qué hace | Script |
|---|---|---|
| 0 | Elige los registros con ritmo anotado | `python codigo/seleccion.py` |
| 1 | Carga cada registro por tramos de 30 min | (la usan las demás etapas) |
| 2 a 6 | Espectro, filtrado y ruido, detección de latidos y serie RR, registro por registro | `python codigo/correr_todo.py todos` |
| 7 y 8 | Ventanas de 60 latidos y 10 características | `python codigo/caracteristicas.py` |
| 9 | Estadística: AUC por sujeto, Wilcoxon, Mann-Whitney, Holm | `python codigo/estadistica.py` |
| 9 | Curva ROC, umbral (Youden) en train y F1 en test, por ventana | `python codigo/roc_umbral.py` |

`correr_todo.py` ejecuta, para cada registro y en este orden, `espectro.py`, `preprocesado.py`, `deteccion_qrs.py` y `serie_rr.py`. Corre varios registros en paralelo y tarda unos 25 a 50 minutos para los 98. También se puede correr un registro: `python codigo/correr_todo.py 112`.

## Qué hace cada etapa

- **Etapa 0, selección.** Se queda con los registros que tienen anotado el ritmo. Las etiquetas se traducen a FA (`(AFIB`), no FA (`(N`) y otro (todo lo demás, que no entra en la comparación). El aleteo auricular queda en "otro".
- **Etapa 2, espectro.** Espectro de Welch en 6 tramos de 5 minutos por registro. No hay pico de red en 50 ni 60 Hz, así que **no se aplica filtro notch**. El pozo que se ve en 75 Hz es el espejo del de 50 Hz, causado por el remuestreo de 125 a 200 Hz.
- **Etapa 3, filtrado y ruido.** Pasa-banda de 0,5 a 40 Hz, filtrado hacia adelante y hacia atrás para no correr los picos. Ventanas de 10 s con dos índices de calidad: bSQI (acuerdo entre Pan-Tompkins y xqrs) y kSQI (curtosis). **Una ventana es ruidosa solo si bSQI < 0,8.** El kSQI se calcula pero no decide, porque confundía ondas T altas con ruido.
- **Etapa 4, filtro adaptativo (ALE/NLMS).** Implementado y evaluado, pero **no se integra**: sube el bSQI a costa de perder latidos, y el ruido real es solo el 0,4% de los datos.
- **Etapa 5, latidos.** Detección de picos R con Pan-Tompkins (implementación propia en `comun.py`), validada contra las anotaciones `.qrs` de la base. Esas anotaciones las generó un detector parecido, así que el acuerdo muestra consistencia y no verdad absoluta.
- **Etapa 6, serie RR.** Descarta RR fuera de 0,25 a 2,5 s, RR en ventanas ruidosas y ectópicos aislados (Dash et al., 2009). Los inválidos quedan marcados, no borrados.
- **Etapa 7, ventanas.** 60 latidos, sin superposición, con al menos 90% de RR válidos, una etiqueta por ventana.
- **Etapa 8, características.** media RR, SDNN, RMSSD, nRMSSD, TPR, entropía de Shannon, SampEn, LF, HF y LF/HF (con Lomb-Scargle; AR como comparación).
- **Etapa 9, estadística.** Mediana por sujeto, Wilcoxon pareado (64 sujetos con FA y no FA), Mann-Whitney, corrección de Holm y AUC por sujeto.

## Resultados principales

- 162.797 ventanas de 60 latidos: 39.530 con FA y 123.267 sin FA.
- AUC por sujeto: TPR 0,911, SampEn 0,892, Shannon 0,884, nRMSSD 0,883, HF 0,863. El LF solo no separa (0,49).
- Las conclusiones no cambian con el criterio de ruido ni con el método de espectro.
- Detección por ventana (umbral elegido en train, evaluado en test, partición por sujeto 70/30): mejor característica sola nRMSSD (F1 0,80 en la partición 0; 0,70 ± 0,09 en 10 particiones). La regla combinada de Dash et al. 2009, con sus umbrales publicados, da F1 0,79 ± 0,07 en 10 particiones (rango 0,67 a 0,90).
- Falsos positivos concentrados en pocos registros (033 y 136: ectópicos con patrón repetido; 051: anotación dudosa).

## Scripts de apoyo

| Script | Para qué |
|---|---|
| `auditar_criterio_ruido.py` | Compara bSQI y kSQI contra las anotaciones y justifica `CRITERIO_RUIDO = "bsqi"` |
| `resumen_calidad.py` | Cuenta ventanas ruidosas por registro |
| `evaluar_ale.py` | Evalúa el filtro adaptativo sobre ventanas ruidosas |
| `comparar_ar_lomb.py` | Compara los dos métodos de espectro del RR |
| `comparar_config.py` | Compara con y sin quitar ectópicos, y 60 contra 128 latidos |
| `medir_polaridad_todos.py`, `contar_cambios.py`, `ver_polaridad.py` | Revisan si la marca de Pan-Tompkins salta entre la R y la S |

## Qué se genera

Se guarda en `resultados/`:

- `registros_anotados.csv`, `validacion_qrs.csv`, `caracteristicas.csv`, `estadistica.csv`, `roc_umbral.csv`, `roc_umbral_estabilidad.csv` y los CSV de comparación.
- `figuras/`: espectros, filtrado, ruido, detección de QRS, tacogramas, diagramas de caja y curva ROC.
- Las carpetas `calidad/`, `rpicos/`, `rr/` y `logs/` son intermedias y no se suben a git.

## Limitaciones

- Las anotaciones `.qrs` vienen de un detector parecido al usado.
- El aleteo auricular queda afuera (mejora posible: juntarlo con la FA, como hace el paper de la base).
- Los ectópicos con patrón repetido siguen dando falsos positivos con el TPR.
- La ubicación de la R puede saltar a la S en QRS bifásicos de unos pocos registros; no se corrigió.
- El 90% de RR válidos y los límites de 0,25 y 2,5 s son criterio propio, no de la bibliografía.

## Bibliografía

- Tsutsui et al., 2025. SHDB-AF (PhysioNet).
- Dash et al., 2009. Detección automática de FA con RMSSD, TPR y entropía de Shannon.
- Widrow et al., 1975. Cancelación adaptativa de ruido.
- Kligfield et al., 2007. Recomendaciones de la AHA para el ECG.
- Li et al., 2008. Índices de calidad de señal de ECG.
- de Chazal et al., 2004. Partición de los datos por sujeto (train/test sin repetir pacientes).

## Integrantes

_Completar._
