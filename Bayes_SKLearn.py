import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
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

# 5. Modelo de Naive Bayes
nb_model = GaussianNB()
nb_model.fit(x_train, y_train)

# Predicciones de Naive Bayes
nb_predictions = nb_model.predict(x_test)

# Mostrar la matriz de confusión
print("Resultados de Naive Bayes:")
print(confusion_matrix(y_test, nb_predictions))

# Generar la tabla de probabilidades para Naive Bayes
nb_probabilities = nb_model.predict_proba(x_test)

# Crear un DataFrame con las probabilidades
nb_table = pd.DataFrame(
    nb_probabilities, 
    columns=["Probabilidad No Trabajador", "Probabilidad Trabajador"]
)
nb_table["Predicción"] = nb_model.predict(x_test)
nb_table["Real"] = y_test.reset_index(drop=True)

# Mostrar las primeras filas de la tabla
print("Tabla de Probabilidades (Naive Bayes):")
print(nb_table.head(10))

# Guardar la tabla como archivo Excel
nb_table.to_excel("tabla_probabilidades_naive_bayes.xlsx", index=False)
print("Tabla guardada como 'tabla_probabilidades_naive_bayes.xlsx'")