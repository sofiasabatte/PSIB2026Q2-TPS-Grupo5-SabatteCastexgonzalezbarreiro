# =============================================================================
# Lo usan todas. Si alguna cambia un parametro, avisar a las demas.
# Pili: PASABANDA_HZ, NOTCH_HZ, UMBRAL_BSQI, UMBRAL_KSQI, CRITERIO_RUIDO
# Juani: TOL_QRS_S, REGISTROS, N_PROCESOS
# Sofi: RR_MIN_S, RR_MAX_S, QUITAR_ECTOPICOS, VENTANA_LATIDOS, MIN_FRACCION_VALIDA
# =============================================================================
# CONFIGURACION COMUN A TODAS LAS ETAPAS
#
# Todos los scripts leen sus parametros de aca, asi se cambian en un solo lugar.
# Las rutas son relativas a la carpeta principal del repo (correr siempre desde ahi).

# --- Datos -----------------------------------------------------------------
DATA_DIR = "data/shdb-af"           # carpeta con 001.hea, 001.dat, 001.atr, 001.qrs, ...
PN_DIR = "shdb-af/1.0.1"     # si un registro no esta en DATA_DIR, se lee directo de PhysioNet
FS = 200                     # Hz. Ojo: grabado a 125 Hz y remuestreado a 200 Hz (Tsutsui et al. 2025)
CANAL = 0                    # 0 = ECG1, 1 = ECG2. Se procesa un solo canal.

# Registros que se EXCLUYEN del analisis estadistico (estadistica.py, roc_umbral.py),
# con el motivo. 005 y 020 tienen el mismo .dat (mismo md5) y son del mismo sujeto
# (4899921); se saca uno. Lo consultamos con la profesora (Vicky).
REGISTROS_EXCLUIDOS = {"020": "duplicado exacto de 005 (mismo md5, mismo sujeto 4899921)"}

# Registros a procesar si no se pasan por linea de comandos.
# "todos" = los 98 registros anotados de la etapa 0 (resultados/registros_anotados.csv).
# Para probar con pocos: REGISTROS = ["001", "002"]
REGISTROS = "todos"
N_PROCESOS = None            # registros en paralelo en correr_todo.py; None = nucleos de la compu - 1

# --- Salidas ---------------------------------------------------------------
RES = "resultados"
FIG = "resultados/figuras"

# --- Carga por tramos (etapa 1) ---------------------------------------------
TRAMO_MIN = 30               # se lee el registro de a 30 min
SOLAPE_S = 5                 # solape entre tramos para no perder latidos en los bordes

# --- Espectro (etapa 2) ----------------------------------------------------
WELCH_SEG_S = 10             # segmento de Welch -> resolucion 0,1 Hz
N_TRAMOS_ESPECTRO = 6        # tramos de 5 min repartidos en las 24 h

# --- Preprocesado (etapa 3) ------------------------------------------------
PASABANDA_HZ = (0.5, 40)     # 0,5 Hz: deriva (AHA, Kligfield 2007). 40 Hz: EMG / alta frecuencia
NOTCH_HZ = None              # poner 50 o 60 SOLO si el espectro (etapa 2) muestra el pico de red
VENTANA_CALIDAD_S = 10       # largo de ventana para los indices de calidad
UMBRAL_BSQI = 0.8            # bSQI = acuerdo Pan-Tompkins vs xqrs (Li et al. 2008). Valor inicial
UMBRAL_KSQI = 5              # curtosis del ECG; ECG limpio > 5 (Li et al. 2008). Valor inicial

# Criterio que decide si una ventana de 10 s es ruidosa.
#   "bsqi"      -> solo bSQI < UMBRAL_BSQI (criterio del paper de la base SHDB-AF)
#   "bsqi_ksqi" -> bSQI o kSQI (criterio anterior, queda para comparar)
# Se eligió "bsqi" porque en la auditoría (ver resumen_calidad.py) las ventanas
# marcadas solo por kSQI se detectan igual de bien que las limpias: eran ondas T altas,
# no ruido.
CRITERIO_RUIDO = "bsqi"

# --- Deteccion de QRS (etapa 5) --------------------------------------------
TOL_QRS_S = 0.150            # tolerancia para emparejar latidos (ANSI/AAMI EC57)

# --- Serie RR (etapa 6) ----------------------------------------------------
RR_MIN_S = 0.25              # 240 lpm. Mas corto = falso QRS
RR_MAX_S = 2.5               # 24 lpm. Mas largo = QRS perdido
QUITAR_ECTOPICOS = True      # ectopicos aislados, criterio de Dash et al. 2009

# --- Ventanas y caracteristicas (etapas 7 y 8) -----------------------------
VENTANA_LATIDOS = 60         # benchmark de SHDB-AF usa 60 latidos; Dash et al. 2009 usa 128
MIN_FRACCION_VALIDA = 0.9    # se descarta la ventana si mas del 10% de sus RR son invalidos
