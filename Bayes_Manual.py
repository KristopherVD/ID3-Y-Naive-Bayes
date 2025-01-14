import pandas as pd
import numpy as np

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

# Dividir los datos en conjuntos de entrenamiento y prueba (80%-20%)
tamano_entrenamiento = int(len(datos) * 0.8)
datos_entrenamiento = datos[:tamano_entrenamiento]
datos_prueba = datos[tamano_entrenamiento:]

# Calcular probabilidades previas
p_trabajador = (datos_entrenamiento["ocupacion"] == "trabajador").mean()
p_no_trabajador = (datos_entrenamiento["ocupacion"] == "no_trabajador").mean()

# Calcular probabilidades condicionales
prob_condicionales = {}
for clase in ["trabajador", "no_trabajador"]:
    subset = datos_entrenamiento[datos_entrenamiento["ocupacion"] == clase]
    prob_condicionales[clase] = {
        "sexo": subset["sexo"].value_counts(normalize=True).to_dict(),
        "causa_def": subset["causa_def"].value_counts(normalize=True).to_dict(),
    }

# Función para predecir la clase de un registro
def predecir_clase(registro):
    prob_trabajador = p_trabajador
    prob_no_trabajador = p_no_trabajador

    for caracteristica, valor in registro.items():
        prob_trabajador *= prob_condicionales["trabajador"].get(caracteristica, {}).get(valor, 1e-6)
        prob_no_trabajador *= prob_condicionales["no_trabajador"].get(caracteristica, {}).get(valor, 1e-6)

    return "trabajador" if prob_trabajador > prob_no_trabajador else "no_trabajador"

# Evaluar el modelo en el conjunto de prueba
predicciones = []
for _, fila in datos_prueba.iterrows():
    predicciones.append(predecir_clase(fila[["sexo", "causa_def"]]))

# Calcular precisión manualmente
predicciones = pd.Series(predicciones, index=datos_prueba.index)
precision = (predicciones == datos_prueba["ocupacion"]).mean()

print(f"\nPrecisión del modelo: {precision:.2%}")

# Inicializar contadores
correctos_trabajador = 0
correctos_no_trabajador = 0
incorrectos_trabajador = 0
incorrectos_no_trabajador = 0

# Iterar sobre los datos de prueba
for _, fila in datos_prueba.iterrows():
    verdadera_clase = fila["ocupacion"]
    prediccion = predecir_clase(fila[["sexo", "causa_def"]])
    
    # Contar los casos según clasificación correcta o incorrecta
    if verdadera_clase == "trabajador":
        if prediccion == "trabajador":
            correctos_trabajador += 1
        else:
            incorrectos_trabajador += 1
    elif verdadera_clase == "no_trabajador":
        if prediccion == "no_trabajador":
            correctos_no_trabajador += 1
        else:
            incorrectos_no_trabajador += 1

# Calcular precisión
total_correctos = correctos_trabajador + correctos_no_trabajador
total_casos = len(datos_prueba)
precision = total_correctos / total_casos * 100

# Mostrar resultados
print("\nClasificación de casos:")
print(f"- Correctos (trabajador): {correctos_trabajador}")
print(f"- Correctos (no trabajador): {correctos_no_trabajador}")
print(f"- Incorrectos (trabajador): {incorrectos_trabajador}")
print(f"- Incorrectos (no trabajador): {incorrectos_no_trabajador}")
