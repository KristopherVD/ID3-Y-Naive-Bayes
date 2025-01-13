import pandas as pd
import numpy as np

def calcular_entropia(datos, columna_etiqueta):
    valores, cuentas = np.unique(datos[columna_etiqueta], return_counts=True)
    probabilidades = cuentas / len(datos)
    return -np.sum(probabilidades * np.log2(probabilidades))

def calcular_ganancia(datos, columna, columna_etiqueta):
    valores, cuentas = np.unique(datos[columna], return_counts=True)
    entropia_inicial = calcular_entropia(datos, columna_etiqueta)
    entropia_condicional = np.sum((cuentas / len(datos)) * np.array([
        calcular_entropia(datos[datos[columna] == valor], columna_etiqueta)
        for valor in valores
    ]))
    return entropia_inicial - entropia_condicional

def construir_arbol(datos, columna_etiqueta):
    if datos.empty or columna_etiqueta not in datos.columns:
        print("Datos vacíos o columna no encontrada.")
        return None

    if datos[columna_etiqueta].nunique() == 1:
        return datos[columna_etiqueta].iloc[0]

    if len(datos.columns) == 1:
        return datos[columna_etiqueta].mode()[0]

    columnas = datos.columns.drop(columna_etiqueta)
    mejor_columna = max(columnas, key=lambda col: calcular_ganancia(datos, col, columna_etiqueta))
    print("Mejor columna seleccionada:", mejor_columna)

    arbol = {mejor_columna: {}}
    for valor in datos[mejor_columna].unique():
        subdatos = datos[datos[mejor_columna] == valor].drop(columns=mejor_columna)
        arbol[mejor_columna][valor] = construir_arbol(subdatos, columna_etiqueta)

    return arbol

# Carga de datos
ruta_archivo = "C:/Users/kvela/Desktop/Muertes totales de trabajadores y no trabajadores por Covid_19 en 2020.xlsx"
datos = pd.read_excel(ruta_archivo)

# Preprocesar columnas
if 'causa_def' in datos.columns:
    datos = datos[['sexo', 'ocupacion', 'causa_def']].dropna()
    datos['sexo'] = datos['sexo'].astype('category')
    datos['ocupacion'] = datos['ocupacion'].astype('category')
    datos['causa_def'] = datos['causa_def'].astype('category')
    
    arbol = construir_arbol(datos, 'causa_def')
    print("Árbol de decisión construido:", arbol)
else:
    print("Error: La columna 'causa_def' no se encuentra en los datos.")