import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix

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

# Codificar variables categóricas con LabelEncoder
le_sexo = LabelEncoder()
le_causa = LabelEncoder()
le_ocupacion = LabelEncoder()

datos["sexo_cod"] = le_sexo.fit_transform(datos["sexo"])
datos["causa_cod"] = le_causa.fit_transform(datos["causa_def"])
datos["ocupacion_cod"] = le_ocupacion.fit_transform(datos["ocupacion"])

# Separar características (X) y etiqueta (y)
X = datos[["sexo_cod", "causa_cod"]]
y = datos["ocupacion_cod"]

# Dividir los datos en conjuntos de entrenamiento y prueba (80%-20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Crear el modelo de Naive Bayes
modelo = MultinomialNB()

# Entrenar el modelo
modelo.fit(X_train, y_train)

# Predecir en el conjunto de prueba
y_pred = modelo.predict(X_test)

# Calcular precisión
precision = accuracy_score(y_test, y_pred)

print(f"\nPrecisión del modelo (scikit-learn): {precision:.2%}")

# Matriz de confusión
matriz_confusion = confusion_matrix(y_test, y_pred)
print("\nMatriz de confusión:")
print(matriz_confusion)

# Resultados detallados
correctos_trabajador = matriz_confusion[0, 0]
correctos_no_trabajador = matriz_confusion[1, 1]
incorrectos_trabajador = matriz_confusion[0, 1]
incorrectos_no_trabajador = matriz_confusion[1, 0]

print("\nClasificación de casos:")
print(f"- Correctos (trabajador): {correctos_trabajador}")
print(f"- Correctos (no trabajador): {correctos_no_trabajador}")
print(f"- Incorrectos (trabajador): {incorrectos_trabajador}")
print(f"- Incorrectos (no trabajador): {incorrectos_no_trabajador}")
