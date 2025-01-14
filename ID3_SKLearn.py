import pandas as pd
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.model_selection import train_test_split

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
X = datos[["sexo", "causa_def"]]  # Características
y = datos["ocupacion"]  # Etiquetas

# Codificar las variables categóricas de las características
X = X.apply(lambda col: col.astype('category').cat.codes)

# Dividir el dataset en conjunto de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Crear el modelo de árbol de decisión (ID3 en sklearn, al usar la entropía como criterio)
modelo = DecisionTreeClassifier(criterion="entropy", random_state=42)

# Entrenar el modelo
modelo.fit(X_train, y_train)

# Evaluar el modelo en el conjunto de prueba
precision = modelo.score(X_test, y_test)
print(f"Precisión del árbol de decisión ID3 con sklearn: {precision:.2%}")

# Generar y mostrar el árbol de decisión como texto
arbol_texto = export_text(modelo, feature_names=["sexo", "causa_def"])
print("\nÁrbol de decisión ID3 (con sklearn):")
print(arbol_texto)
