import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import confusion_matrix

# 1. Función para cargar los datos
def load_data(file_path):
    excel_data = pd.ExcelFile(file_path)
    sheets_data = {sheet: excel_data.parse(sheet) for sheet in excel_data.sheet_names}
    return sheets_data

# 2. Función para preprocesar los datos
def preprocess_data(data):
    data = data[['sexo', 'ocupacion', 'causa_def']].dropna()
    data['trabajador'] = data['ocupacion'].apply(lambda x: 1 if x == 4 else 0)
    features = pd.get_dummies(data[['sexo', 'causa_def']], drop_first=True)
    labels = data['trabajador']
    return features, labels

# 3. Cargar y procesar los datos
file_path = "C:/Users/kvela/Desktop/Muertes totales de trabajadores y no trabajadores por Covid_19 en 2020.xlsx"
sheets_data = load_data(file_path)
data = sheets_data['Muertes_Totales']

features, labels = preprocess_data(data)

# 4. Dividir los datos en entrenamiento y prueba
x_train, x_test, y_train, y_test = train_test_split(features, labels, test_size=0.3, random_state=42)

# 5. Modelo de Árbol de Decisión (ID3)
id3_model = DecisionTreeClassifier(criterion='entropy', random_state=42)
id3_model.fit(x_train, y_train)

# Predicciones del Árbol de Decisión
id3_predictions = id3_model.predict(x_test)

# Mostrar la matriz de confusión
print("Resultados del Árbol de Decisión (ID3):")
print(confusion_matrix(y_test, id3_predictions))

# Graficar el Árbol de Decisión
plt.figure(figsize=(20, 10))
plot_tree(
    id3_model, 
    feature_names=features.columns, 
    class_names=["No Trabajador", "Trabajador"], 
    filled=True, 
    rounded=True
)
plt.title("Árbol de Decisión (ID3)")
plt.show()