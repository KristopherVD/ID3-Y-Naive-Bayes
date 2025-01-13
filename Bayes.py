import pandas as pd
import numpy as np

def load_data(file_path):
    excel_data = pd.ExcelFile(file_path)
    sheets_data = {sheet: excel_data.parse(sheet) for sheet in excel_data.sheet_names}
    return sheets_data

def preprocess_data(data):
    # Filtrar las columnas relevantes
    data = data[['causa_def', 'sexo', 'ocupacion']].dropna()
    
    features = data[['sexo', 'ocupacion']]
    labels = data['causa_def']
    
    # Convertir variables categóricas en variables dummy
    features = pd.get_dummies(features, drop_first=True)
    
    return features, labels

def train_naive_bayes(features, labels):
    # Calcular las probabilidades previas
    prior_prob = labels.value_counts(normalize=True)
    
    # Calcular las probabilidades condicionales
    conditional_prob = {}
    for column in features.columns:
        conditional_prob[column] = {}
        for value in features[column].unique():
            conditional_prob[column][value] = {}
            for label in labels.unique():
                conditional_prob[column][value][label] = (
                    ((features[column] == value) & (labels == label)).sum() + 1) / ((labels == label).sum() + len(features[column].unique()))

    return prior_prob, conditional_prob

def predict_naive_bayes(features, prior_prob, conditional_prob):
    predictions = []
    for _, row in features.iterrows():
        label_prob = {}
        for label in prior_prob.index:
            label_prob[label] = prior_prob[label]
            for column in features.columns:
                label_prob[label] *= conditional_prob[column].get(row[column], {}).get(label, 1 / (len(features[column].unique()) + 1))
        predictions.append(max(label_prob, key=label_prob.get))
    return predictions

file_path = 'C:/Users/kvela/Desktop/Muertes totales de trabajadores y no trabajadores por Covid_19 en 2020.xlsx'
sheets_data = load_data(file_path)
all_results = {}

for sheet_name, data in sheets_data.items():
    try:
        print(f"Procesando hoja: {sheet_name}")
        features, labels = preprocess_data(data)
        print(f"Características:\n{features.head()}")
        print(f"Etiquetas:\n{labels.head()}")
        prior_prob, conditional_prob = train_naive_bayes(features, labels)
        predictions = predict_naive_bayes(features, prior_prob, conditional_prob)
        accuracy = (predictions == labels).mean()
        all_results[sheet_name] = {"accuracy": accuracy}
    except Exception as e:
        all_results[sheet_name] = {"error": str(e)}

print(all_results)