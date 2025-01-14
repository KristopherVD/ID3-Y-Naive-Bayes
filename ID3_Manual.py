import pandas as pd
import numpy as np
from math import log2

# Cargar el dataset desde la ruta proporcionada
ruta_archivo = "C:/Users/kvela/Documents/Ux/3er semestre/Estructuras de datos/Tercer parcial/Examen/Muertes totales de trabajadores y no trabajadores por Covid_19 en 2020.xlsx"
datos = pd.read_excel(ruta_archivo)

# Filtrar datos relevantes (sexo, ocupación, causa de defunción)
datos = datos[["sexo", "ocupacion", "causa_def"]]

# Filtrar únicamente los casos válidos para "ocupación" y "causa de defunción"
datos = datos[(datos["ocupacion"].isin([4, 11])) & (datos["causa_def"].isin(["U071", "U072"]))]

# Codificar las columnas categóricas
mapa_sexo = {1: "hombre", 2: "mujer", 9: "no_especificado"}
mapa_ocupacion = {4: "trabajador", 11: "no_trabajador"}
mapa_causa = {"U071": "covid_diagnosticado", "U072": "covid_postmortem"}

datos["sexo"] = datos["sexo"].map(mapa_sexo)
datos["ocupacion"] = datos["ocupacion"].map(mapa_ocupacion)
datos["causa_def"] = datos["causa_def"].map(mapa_causa)

# Función para calcular la entropía
def calcular_entropia(columna):
    valores = columna.value_counts(normalize=True)
    return -sum(valores * np.log2(valores))

# Función para calcular la ganancia de información
def ganancia_informacion(datos, atributo, clase):
    # Calcular la entropía total (para la clase)
    entropia_total = calcular_entropia(datos[clase])
    
    # Calcular la ganancia de información
    valores_atributo = datos[atributo].unique()
    ganancia = entropia_total
    for valor in valores_atributo:
        # Filtrar los datos para cada valor del atributo
        datos_filtrados = datos[datos[atributo] == valor]
        probabilidad = len(datos_filtrados) / len(datos)
        ganancia -= probabilidad * calcular_entropia(datos_filtrados[clase])
    
    return ganancia

# Función recursiva para construir el árbol de decisión (ID3)
def construir_arbol(datos, atributos, clase):
    # Si todos los registros tienen la misma clase, devolver la clase
    if len(datos[clase].unique()) == 1:
        return datos[clase].iloc[0]
    
    # Si no hay más atributos para dividir, devolver la clase mayoritaria
    if not atributos:
        return datos[clase].mode()[0]
    
    # Seleccionar el atributo con la mayor ganancia de información
    mejor_atributo = max(atributos, key=lambda attr: ganancia_informacion(datos, attr, clase))
    
    # Crear el nodo del árbol
    arbol = {mejor_atributo: {}}
    
    # Recursión para cada valor del mejor atributo
    for valor in datos[mejor_atributo].unique():
        # Filtrar los datos para este valor del atributo
        datos_filtrados = datos[datos[mejor_atributo] == valor]
        
        # Llamar recursivamente para construir el subárbol
        subarbol = construir_arbol(datos_filtrados, [attr for attr in atributos if attr != mejor_atributo], clase)
        arbol[mejor_atributo][valor] = subarbol
    
    return arbol

# Definir los atributos y la clase
atributos = ["sexo", "causa_def"]
clase = "ocupacion"

# Construir el árbol
arbol_decision = construir_arbol(datos, atributos, clase)

# Mostrar el árbol
print("\nÁrbol de decisión ID3 (manual):")
print(arbol_decision)

# Función para hacer predicciones con el árbol
def predecir_con_arbol(arbol, registro):
    if isinstance(arbol, dict):
        atributo = list(arbol.keys())[0]
        valor = registro[atributo]
        return predecir_con_arbol(arbol[atributo].get(valor, 'Desconocido'), registro)
    return arbol

# Evaluar el modelo
predicciones = []
for _, fila in datos.iterrows():
    predicciones.append(predecir_con_arbol(arbol_decision, fila))

# Calcular precisión manualmente
predicciones = pd.Series(predicciones, index=datos.index)
precision_manual = (predicciones == datos["ocupacion"]).mean()
print(f"\nPrecisión del árbol de decisión manual: {precision_manual:.2%}")
