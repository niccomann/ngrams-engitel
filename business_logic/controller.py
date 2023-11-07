import os
from flask import Flask, jsonify, request
from flask_cors import CORS
from business_logic.bigquery import get_bodies_from_bigquery
from business_logic.kmeans_on_bigquery import ottieni_table_generata_com_kmeans
from business_logic.model_with_pandas import predict_sqli_pandas, predict_xss_pandas, predict_all_pandas
from business_logic.model_with_tf_idf import predict_sqli_tf_idf, predict_xss_tf_idf
from llama.privateGPT import evaluate_prompt
import pandas as pd

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

app = Flask(__name__)

cors = CORS(app, resources={r"/*": {"origins": "*"}})


# Esempio di endpoint GET
@app.route('/get_bodies')
def get_bodies():
    print("Endpoint 'get_bodies' called")
    list_of_requests = get_bodies_from_bigquery()
    list_of_requests = list_of_requests[:6]
    return jsonify(list_of_requests)


@app.route('/model_generic_attack', methods=['GET'])
def model_generic_attack():
    print('inizio model_sqli_pandas')
    corpo_richiesta = request.args.get('corpo_richiesta')
    y = predict_all_pandas(corpo_richiesta)
    print(y[1])
    if y[1] == '1':
        y = 'attack detected'
    else:
        y = 'attack not detected'
    return jsonify(y)



@app.route('/model_sqli_endpoint', methods=['GET'])
def model_sqli():
    print('inizio model_sqli_pandas')
    corpo_richiesta = request.args.get('corpo_richiesta')
    y = predict_sqli_pandas(corpo_richiesta)
    print(y[0])

    if y[1] == '1':
        y = 'SQLi detected'
    else:
        y = 'SQLi not detected'
    return jsonify(y)


@app.route('/model_sqli_endpoint_tfidf', methods=['GET'])
def model_sqli_tfidf():
    print('inizio model_sqli_tfidf')
    corpo_richiesta = request.args.get('corpo_richiesta')
    y = predict_sqli_tf_idf(corpo_richiesta)
    print(y[0])

    if y[1] == '1':
        y = 'SQLi detected'
    else:
        y = 'SQLi not detected'
    return jsonify(y)


@app.route('/model_XSS_endpoint', methods=['GET'])
def model_XSS():
    corpo_richiesta = request.args.get('corpo_richiesta')
    y = predict_xss_pandas(corpo_richiesta)
    print(y[0])

    if y[1] == '1':
        y = 'xss detected'
    else:
        y = 'xss not detected'
    return jsonify(y)


@app.route('/model_XSS_endpoint_tfidf', methods=['GET'])
def model_XSS_tfidf():
    corpo_richiesta = request.args.get('corpo_richiesta')
    y = predict_xss_tf_idf(corpo_richiesta)
    print(y[0])

    if y[1] == '1':
        y = 'xss detected'
    else:
        y = 'xss not detected'
    return jsonify(y)



@app.route('/kmeans_unsupervised', methods=['GET'])
def kmeans_model():
    print('kmenas chiamato')
    df = ottieni_table_generata_com_kmeans()
    df_json = df.to_json(orient='values')
    return jsonify(df_json)


@app.route('/api/endpoints')
def get_endpoints():
    endpoints = []
    for rule in app.url_map.iter_rules():
        if "static" not in rule.endpoint and "error" not in rule.endpoint:
            endpoints.append(rule.rule)
    return jsonify(endpoints)

@app.route('/all_models', methods=['GET'])
def all_models():
    corpo_richiesta = request.args.get('corpo_richiesta')
    y = 'xss_tfidf:' + predict_xss_tf_idf(corpo_richiesta)
    y = 'sqli_tfidf' + y + predict_sqli_tf_idf(corpo_richiesta)
    y = 'xss_pandas' + y + predict_xss_pandas(corpo_richiesta)
    y = 'sqli_pandas' + y + predict_sqli_pandas(corpo_richiesta)
    os.chdir(ROOT_DIR + '/llama')
    answer, other_info = evaluate_prompt(
        "Puoi dirmi se la seguente richiesta http: " + corpo_richiesta + " è SQLi, rispondimi si o no e forniscimi una spiegazione?")
    y = y + answer
    return jsonify(y)
@app.errorhandler(Exception)
def handle_error(error):
    error_message = str(error)
    print("Errore:", error_message)
    return error_message, 500


if __name__ == '__main__':
    app.run()
