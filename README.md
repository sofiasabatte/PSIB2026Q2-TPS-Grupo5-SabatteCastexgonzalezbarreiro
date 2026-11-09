# Detección de fibrilación auricular con HRV (SHDB-AF)

TP de Procesamiento de Señales e Imágenes Biomédicas (16.63, ITBA), 2026 Q2. Grupo 5: Sofía Sabatte, Juana Castex y María del Pilar González Barreiro.

## De qué trata

La fibrilación auricular (FA) hace que el ritmo del corazón sea irregular. Este trabajo la detecta **mirando solo el ritmo**: calcula 10 medidas sobre los tiempos entre latidos (RR) en ventanas de 60 latidos y mide qué tan bien separa cada una la FA del ritmo no-FA. No usa machine learning: cada característica se evalúa con un umbral, y además se prueba la regla combinada de Dash et al. (2009).

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
| 9 | Curva ROC, umbral (Youden) en train y Se, Esp y F1 en test, por ventana | `python codigo/roc_umbral.py` |

`correr_todo.py` ejecuta, para cada registro y en este orden, `espectro.py`, `preprocesado.py`, `deteccion_qrs.py` y `serie_rr.py`. Corre varios registros en paralelo y tarda unos 25 a 50 minutos para los 98. También se puede correr un registro: `python codigo/correr_todo.py 112`.

`estadistica.py` y `roc_umbral.py` solo necesitan `resultados/caracteristicas.csv`, que está en el repo, así que se pueden correr sin bajar la base.

## Qué hace cada etapa

- **Etapa 0, selección.** Se queda con los registros que tienen anotado el ritmo. Las etiquetas se traducen a FA (`(AFIB`), no-FA (`(N`) y otro (todo lo demás, que no entra en la comparación). En esta base `(N` significa "no anotado", así que no-FA incluye el ritmo sinusal y cualquier otro ritmo sin etiqueta. El aleteo auricular queda en "otro". Ningún registro anotado tiene marcapasos durante el Holter, así que ese filtro no excluye ninguno.
- **Etapa 2, espectro.** Espectro de Welch en 6 tramos de 5 minutos por registro. La señal ya viene filtrada por los autores de la base, y el espectro lo confirma en los 97 registros:
  - en 50 Hz hay un pozo y no un pico (la potencia vale entre 0,05 y 0,52 veces la de su vecindad), o sea que el Holter ya aplicó un notch de red, y **no se agrega otro**;
  - el pozo en 75 Hz aparece en todos (entre 0,04 y 0,44 veces su vecindad). Es el espejo del de 50 Hz causado por el remuestreo de 125 a 200 Hz (125 − 50 = 75). El espectro entre 62,5 y 100 Hz copia al de 25 a 62,5 Hz (correlación mediana 0,95; en 95 de 97 registros es mayor a 0,8, y en los dos restantes, 037 y 077, los pozos se ven igual pero la banda alta queda contra el piso de ruido). Lo que está por encima de 62,5 Hz no es información real.
- **Etapa 3, filtrado y ruido.** Pasa-banda de 0,5 a 40 Hz con fase cero (filtrado hacia adelante y hacia atrás). No es para limpiar la señal, que ya viene filtrada: el corte de 0,5 Hz no cambia nada sobre una señal que ya trae un pasa-altos de 0,67 Hz, y el de 40 Hz deja afuera la imagen del remuestreo por encima de 62,5 Hz. El filtro acondiciona la señal que usan los dos detectores del índice de calidad y la ubicación de la R. Después, ventanas de 10 s con dos índices de calidad: bSQI (acuerdo entre Pan-Tompkins y xqrs) y kSQI (curtosis). **Una ventana es ruidosa solo si bSQI < 0,8**, y así se excluye el 0,41 % de las ventanas (los registros más ruidosos son el 131, con 13,5 %, el 021, con 7,6 %, y el 113, con 6,3 %). El kSQI se calcula pero no decide: en la auditoría (`auditar_criterio_ruido.py`), las ventanas que marca solo el kSQI se detectan casi igual que las limpias (Se 0,989 y VPP 0,992, contra 0,999 y 0,999), porque eran ondas T altas y no ruido.
- **Etapa 4, filtro adaptativo (ALE/NLMS).** Implementado y evaluado con 128 combinaciones de parámetros sobre ventanas ruidosas reales de 10 registros, en una tanda exploratoria y otra de confirmación. **No se integra**: sube el bSQI, pero la sensibilidad contra las anotaciones baja, es decir que los dos detectores pasan a coincidir sobre latidos equivocados. Como el ruido real es solo el 0,41 % de los datos, esas ventanas se excluyen.
- **Etapa 5, latidos.** Detección de picos R con Pan-Tompkins (implementación propia en `comun.py`, con los umbrales adaptativos y la búsqueda hacia atrás de Hamilton y Tompkins), validada contra las anotaciones `.qrs` de la base: Se 99,83 % y VPP 99,76 % en total, 99,91 % y 99,91 % en ventanas limpias. Esas anotaciones las generó un detector parecido, así que el acuerdo muestra consistencia y no exactitud.
- **Etapa 6, serie RR.** Descarta RR fuera de 0,25 a 2,5 s, RR en ventanas ruidosas y ectópicos aislados (Dash et al., 2009). Los inválidos quedan marcados, no borrados. No se eliminan latidos por irregularidad, porque en FA la irregularidad es justamente lo que se mide.
- **Etapa 7, ventanas.** 60 latidos, sin superposición, con al menos 90 % de RR válidos, una etiqueta por ventana.
- **Etapa 8, características.** Media RR, SDNN, RMSSD, nRMSSD, TPR, entropía de Shannon, SampEn, LF, HF y LF/HF (con el periodograma de Lomb; AR de Yule-Walker con orden elegido por AIC como comparación).
- **Etapa 9, estadística y detección.** Mediana por sujeto, Wilcoxon pareado (64 sujetos con FA y no-FA), Mann-Whitney, corrección de Holm y AUC por sujeto. Después, detección por ventana: partición por sujeto 70/30 (65 sujetos en train y 28 en test), umbral de Youden elegido en train y evaluado en test, repetido con 10 particiones distintas.

## Resultados principales

- 162.797 ventanas de 60 latidos: 39.530 con FA y 123.267 sin FA.
- AUC por sujeto: TPR 0,911, SampEn 0,892, Shannon 0,884, nRMSSD 0,883, HF 0,863. El LF solo no separa (0,49).
- En FA, las potencias relativas de LF y HF (0,21 y 0,51) y el LF/HF (0,42) son casi las de un espectro plano entre 0,01 y 0,5 Hz, donde cada banda se lleva una parte proporcional a su ancho. La serie RR en FA se comporta como ruido blanco, y el LF/HF pierde su lectura como balance simpatovagal.
- Detección por ventana, en 10 particiones por sujeto (media ± desvío del F1 en test):
  - la regla combinada de Dash et al. (2009), con sus umbrales publicados y sin ajustar nada, da 0,79 ± 0,07 (de 0,67 a 0,90) y es la mejor en 9 de las 10 particiones;
  - la mejor característica sola es nRMSSD, con 0,70 ± 0,09 (de 0,57 a 0,80).
  El F1 varía bastante entre particiones porque cambia la proporción de FA en test según qué sujetos caen ahí (de 7.900 a 14.400 ventanas con FA). La sensibilidad y la especificidad de nRMSSD son más estables (Se entre 0,85 y 0,99, Esp entre 0,73 y 0,86).
- Las AUC por ventana (`roc_umbral.py`) son algo menores que las AUC por sujeto (`estadistica.py`), porque la mediana por sujeto suaviza la variación entre ventanas.

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

- `registros_anotados.csv`, `espectro_resumen.csv`, `validacion_qrs.csv`, `resumen_calidad.csv`, `auditoria_ruido.csv`, `polaridad_r.csv`, `caracteristicas.csv`, `estadistica.csv`, `roc_umbral.csv`, `roc_umbral_estabilidad.csv` y los CSV de comparación.
- `figuras/`: espectros, filtrado, ruido, resumen de calidad, auditoría del criterio de ruido, detección de QRS, tacogramas, diagramas de caja y curva ROC.
- Las carpetas `calidad/`, `rpicos/`, `rr/` y `logs/` son intermedias y no se suben a git.

## Limitaciones

- Las anotaciones `.qrs` vienen de un detector parecido al usado, así que la validación mide acuerdo entre detectores y no exactitud.
- No-FA mezcla el ritmo sinusal con cualquier ritmo no anotado, y el aleteo auricular queda afuera (mejora posible: juntarlo con la FA, como hace el paper de la base). La base es de una sola institución.
- Los ectópicos frecuentes (por ejemplo, el bigeminismo) también hacen irregular la serie RR y pueden dar falsos positivos.
- En QRS bifásicos la marca de la R puede saltar a la S. Se midió en los 97 registros (`polaridad_r.csv`, 5 minutos del medio de cada uno): solo el 021 y el 033 tienen las marcas mezcladas, los dos en FA y con RMSSD de 280 a 375 ms, y la diferencia mediana de RMSSD al fijar la polaridad es 0 ms. Por eso no se corrigió.
- Con ventanas de 60 latidos (unos 40 a 50 s) la banda LF tiene pocos ciclos y su estimación es poco confiable.
- Los umbrales de Dash se publicaron para ventanas de 128 latidos y acá se aplican a ventanas de 60.
- El 90 % de RR válidos y los límites de 0,25 y 2,5 s son criterio propio, no de la bibliografía.

## Bibliografía

- Tsutsui, K., Biton Brimer, S. y Behar, J. (2025). SHDB-AF (versión 1.0.1). PhysioNet. https://doi.org/10.13026/n6yq-fq90
- Tsutsui, K. et al. (2025). SHDB-AF: a Japanese Holter ECG database of atrial fibrillation. *Scientific Data*, 12, 454. https://doi.org/10.1038/s41597-025-04777-4
- Pan, J. y Tompkins, W. J. (1985). A real-time QRS detection algorithm. *IEEE Trans. Biomed. Eng.*, 32(3), 230-236.
- Hamilton, P. S. y Tompkins, W. J. (1986). Quantitative investigation of QRS detection rules using the MIT/BIH arrhythmia database. *IEEE Trans. Biomed. Eng.*, 33(12), 1157-1165.
- Kligfield, P. et al. (2007). Recommendations for the standardization and interpretation of the electrocardiogram: Part I. *Circulation*, 115, 1306-1324.
- Li, Q., Mark, R. G. y Clifford, G. D. (2008). Robust heart rate estimation from multiple asynchronous noisy sources using signal quality indices and a Kalman filter. *Physiol. Meas.*, 29(1), 15-32.
- Widrow, B. et al. (1975). Adaptive noise cancelling: principles and applications. *Proc. IEEE*, 63(12), 1692-1716.
- Fotiadou, E. et al. (2018). Enhancement of low-quality fetal electrocardiogram based on time-sequenced adaptive filtering. *Med. Biol. Eng. Comput.*, 56.
- Dash, S. et al. (2009). Automatic real time detection of atrial fibrillation. *Ann. Biomed. Eng.*, 37(9), 1701-1709.
- Task Force of the ESC and NASPE (1996). Heart rate variability: standards of measurement, physiological interpretation and clinical use. *Circulation*, 93(5), 1043-1065.
- Laguna, P., Moody, G. B. y Mark, R. G. (1998). Power spectral density of unevenly sampled data by least-square analysis. *IEEE Trans. Biomed. Eng.*, 45(6), 698-715.
- Richman, J. S. y Moorman, J. R. (2000). Physiological time-series analysis using approximate entropy and sample entropy. *Am. J. Physiol. Heart Circ. Physiol.*, 278(6), H2039-H2049.
- Boardman, A. et al. (2002). A study on the optimum order of autoregressive models for heart rate variability. *Physiol. Meas.*, 23, 325-336.
- de Chazal, P., O'Dwyer, M. y Reilly, R. B. (2004). Automatic classification of heartbeats using ECG morphology and heartbeat interval features. *IEEE Trans. Biomed. Eng.*, 51(7), 1196-1206.

## Integrantes

- Sofía Sabatte
- Juana Castex
- María del Pilar González Barreiro
