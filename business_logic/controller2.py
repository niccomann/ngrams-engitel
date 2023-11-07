import os
from flask import Flask, jsonify, request
from flask_cors import CORS

from business_logic import pipeline_of_models
from llama.privateGPT import evaluate_prompt

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

app = Flask(__name__)

cors = CORS(app, resources={r"/*": {"origins": "*"}})


# Esempio di endpoint GET
@app.route('/model_gpt4all_endpoint', methods=['GET'])
def model_GPT():
    corpo_richiesta = request.args.get('corpo_richiesta')
    #splitta il corpo della richiesta in due parti con il carattere separatore &
    os.chdir(ROOT_DIR + '/llama')
    answer, other_info = evaluate_prompt(
        "Puoi dirmi se la seguente richiesta http: " + corpo_richiesta + " è SQLi, rispondimi si o no e forniscimi una spiegazione?")
    return jsonify(answer)


@app.route('/model_llama_endpoint', methods=['GET'])
def model_llama():
    corpo_richiesta = request.args.get('corpo_richiesta')
    #splitta il corpo della richiesta in due parti con il carattere separatore &
    os.chdir(ROOT_DIR + '/llama')
    answer, other_info = evaluate_prompt(
        "Puoi dirmi se la seguente richiesta http: " + corpo_richiesta + " è SQLi, rispondimi si o no e forniscimi una spiegazione?")
    return jsonify(answer)

@app.route('/run_llama_only_if_other_models_detect_attack_of_all_request_of_last_week', methods=['GET'])
def all_models():
    y = pipeline_of_models.analyze_requests()
    return jsonify(y)


@app.route('/api/endpoints')
def get_endpoints():
    endpoints = []
    for rule in app.url_map.iter_rules():
        if "static" not in rule.endpoint and "error" not in rule.endpoint:
            endpoints.append(rule.rule)
    return jsonify(endpoints)


if __name__ == '__main__':


    ## run app at port 5001
    app.run(port=5001, debug=True)


