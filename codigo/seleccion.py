# ETAPA 0: seleccion de registros
#
# Que hace: lee la tabla con los datos de los 128 registros (AdditionalData.csv)
# y se queda solo con los que tienen el ritmo anotado (columna "Annotated").
# Por que: sin anotaciones no sabemos cuando hay FA, asi que esos registros
# no sirven para comparar FA contra no-FA.
#
# Entra:  resultados/AdditionalData.csv   (128 registros)
# Sale:   resultados/registros_anotados.csv  (98 registros; 97 registros sin el duplicado 020 (005 y 020 son duplicados), de 93 sujetos distintos, ver REGISTROS_EXCLUIDOS en config.py)
#
# Como correrlo (desde la carpeta principal del repo):
#   python codigo/seleccion.py

import csv                      # para leer y escribir archivos .csv
from collections import Counter  # para contar cuantas veces aparece cada sujeto

# Rutas de los archivos (relativas a la carpeta principal del repo)
ENTRADA = "resultados/AdditionalData.csv"
SALIDA = "resultados/registros_anotados.csv"


def cargar(ruta):
    """Lee el .csv y devuelve una lista de filas; cada fila es un diccionario
    (por ejemplo fila["Data_ID"] da el numero de registro)."""
    with open(ruta, encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def seleccionar(filas):
    """Se queda solo con las filas donde Annotated es 'True'."""
    anotados = []
    for fila in filas:
        # .strip() saca espacios y .lower() pasa a minuscula, por si viene 'True ' o 'TRUE'
        if fila["Annotated"].strip().lower() == "true":
            anotados.append(fila)
    return anotados


def resumen(filas, anotados):
    """Imprime numeros para chequear que la seleccion tiene sentido."""
    # Cuantos registros tiene cada sujeto
    por_sujeto = Counter(fila["Subject_ID"] for fila in anotados)
    # Sujetos con mas de un registro (importante: al separar train/test
    # hay que mantenerlos juntos, por eso guardamos el Subject_ID)
    repetidos = {s: n for s, n in por_sujeto.items() if n > 1}
    # Registros donde el paciente tenia marcapasos durante el Holter
    con_marcapasos = [f["Data_ID"] for f in anotados
                      if f["PPM_on_Holter"].strip().lower() == "true"]

    print("registros totales:       ", len(filas))
    print("registros anotados:      ", len(anotados))
    print("sujetos distintos:       ", len(por_sujeto))
    print("sujetos con >1 registro: ", len(repetidos), repetidos)
    print("con marcapasos:          ", len(con_marcapasos), con_marcapasos)


def guardar(anotados, ruta):
    """Guarda los registros elegidos en un .csv nuevo, con las columnas
    que vamos a usar mas adelante."""
    columnas = ["Data_ID", "Subject_ID", "AF_Type",
                "PPM_on_Holter", "Holter_recording_length"]
    with open(ruta, "w", encoding="utf-8", newline="") as f:
        escritor = csv.DictWriter(f, fieldnames=columnas)
        escritor.writeheader()                      # primera linea: nombres de columnas
        for fila in anotados:
            escritor.writerow({c: fila[c] for c in columnas})


# Esto se ejecuta solo cuando corres el archivo directamente
if __name__ == "__main__":
    filas = cargar(ENTRADA)
    anotados = seleccionar(filas)
    resumen(filas, anotados)
    guardar(anotados, SALIDA)
    print("guardado en", SALIDA)
