import os
from flask import Flask, jsonify, request
from flask_cors import CORS

from business_logic import pipeline_of_models

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

app = Flask(__name__)

cors = CORS(app, resources={r"/*": {"origins": "*"}})




@app.route('/llama_on_last_month_of_http_requests', methods=['GET'])
def model_llama():

    os.chdir(ROOT_DIR + '/llama')
    answer, start_date, end_date = pipeline_of_models.analyze_requests_with_llama_last_month()
    json_to_return = { "answer": answer, "start_date": start_date, "end_date": end_date }
    return jsonify(json_to_return)


@app.route('/api/endpoints')
def get_endpoints():
    endpoints = []
    print('ciao')
    for rule in app.url_map.iter_rules():
        if "static" not in rule.endpoint and "error" not in rule.endpoint:
            endpoints.append(rule.rule)
    return jsonify(endpoints)



if __name__ == '__main__':

    ## run app at port 5001
    app.run(port=5002, debug=True)