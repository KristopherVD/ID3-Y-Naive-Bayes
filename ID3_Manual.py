import pandas as pd
import numpy as np
import os
from math import log2

# Obtener la ruta del directorio actual
directorio_actual = os.path.dirname(os.path.abspath(__file__))

# Construir la ruta del archivo basado en el directorio actual
nombre_archivo = "Muertes totales de trabajadores y no trabajadores por Covid_19 en 2020.xlsx"
ruta_archivo = os.path.join(directorio_actual, nombre_archivo)

# Verificar si el archivo existe
if not os.path.exists(ruta_archivo):
    print(f"Error: No se encontró el archivo '{nombre_archivo}' en la carpeta actual.")
    exit()

# Cargar el archivo de Excel
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

# Dividir los datos en conjunto de entrenamiento (80%) y conjunto de prueba (20%) de manera manual
# Mezclar aleatoriamente los datos
datos_aleatorios = datos.sample(frac=1, random_state=42).reset_index(drop=True)

# Determinar el índice de la división
tamanio_entrenamiento = int(0.8 * len(datos_aleatorios))

# Crear los conjuntos de entrenamiento y prueba
datos_entrenamiento = datos_aleatorios[:tamanio_entrenamiento]
datos_prueba = datos_aleatorios[tamanio_entrenamiento:]

# Separar las características (X) y las etiquetas (y) en ambos conjuntos
X_entrenamiento = datos_entrenamiento[["sexo", "causa_def"]]
y_entrenamiento = datos_entrenamiento["ocupacion"]
X_prueba = datos_prueba[["sexo", "causa_def"]]
y_prueba = datos_prueba["ocupacion"]

# Codificar las variables categóricas de las características
X_entrenamiento = X_entrenamiento.apply(lambda col: col.astype('category').cat.codes)
X_prueba = X_prueba.apply(lambda col: col.astype('category').cat.codes)

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

# Construir el árbol con el conjunto de entrenamiento
arbol_decision = construir_arbol(pd.concat([X_entrenamiento, y_entrenamiento], axis=1), atributos, clase)

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

# Evaluar el modelo en el conjunto de prueba
predicciones = []
for _, fila in X_prueba.iterrows():
    predicciones.append(predecir_con_arbol(arbol_decision, fila))

# Calcular precisión manualmente
predicciones = pd.Series(predicciones, index=X_prueba.index)
precision_manual = (predicciones == y_prueba).mean()
print(f"\nPrecisión del árbol de decisión: {precision_manual:.2%}")
