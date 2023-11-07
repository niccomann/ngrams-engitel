import csv
import os
import numpy as np
import shap
from matplotlib import pyplot as plt
import json

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def create_shapely_values(rfc, X_train_Tfidf, X_test_Tfidf, vectorizer, http_string):
    # Explain the model using shapley values
    explainer = shap.Explainer(rfc, X_train_Tfidf.toarray(), check_additivity=False)
    shap_values = explainer.shap_values(X_test_Tfidf.toarray())

    # Plot the shapley values for the first instance of X_test
    shap.summary_plot(shap_values[0], vectorizer.get_feature_names_out(), plot_type='bar')
    # plt.title(http_string)
    plt.show()
    print('per la prima istanza:')

    np.savetxt('shap_values.csv', shap_values[0], delimiter=',', header=','.join(vectorizer.get_feature_names_out()))

    csv_file_path = ROOT_DIR + '/UI/visualizations/shap_values.csv'
    json_file_path = ROOT_DIR + '/UI/visualizations/shap_values.json'

    # Leggi il file CSV e convertilo in una struttura dati Python
    data = []
    with open(csv_file_path, 'r') as csv_file:
        csv_reader = csv.DictReader(csv_file)
        for row in csv_reader:
            data.append(row)

    # Scrivi i dati come JSON nel file di destinazione
    with open(json_file_path, 'w') as json_file:
        json.dump(data, json_file)

    # Plot dell'interpretazione per la primo istanza di X_test_df
    shap_plot = shap.force_plot(explainer.expected_value[0], shap_values[0], vectorizer.get_feature_names_out(),
                                feature_names=vectorizer.get_feature_names_out(),
                                matplotlib=False)

    # salva shap_plot in un file html
    shap.save_html('../UI/shap_plot_generato_da_fenera_shap_plot_strano.html', shap_plot)
