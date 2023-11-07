import csv
import json
import os
import numpy as np

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def salva_come_json_e_csv(shap_values, feature_names):

    np.savetxt('shap_values.csv', shap_values[0], delimiter=',', header=','.join(feature_names))

    csv_file_path = './UI/visualizations/shap_values.csv'
    json_file_path = './UI/visualizations/shap_values.json'

    # Leggi il file CSV e convertilo in una struttura dati Python
    data = []
    with open(csv_file_path, 'r') as csv_file:
        csv_reader = csv.DictReader(csv_file)
        for row in csv_reader:
            data.append(row)

    # Scrivi i dati come JSON nel file di destinazione
    with open(json_file_path, 'w') as json_file:
        json.dump(data, json_file)


