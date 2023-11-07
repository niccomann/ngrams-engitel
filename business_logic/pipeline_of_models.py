import datetime
import os
import shutil

from business_logic import model_with_pandas, model_with_tf_idf
from business_logic.bigquery import execute_query_on_bigquery
from llama.privateGPT import evaluate_prompt
from llama.ingest import ingest
from textblob import TextBlob

import calendar

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def analizza_sentimento(frase):
    analisi = TextBlob(frase)
    if analisi.sentiment.polarity > 0:
        return "Positivo"
    elif analisi.sentiment.polarity == 0:
        return "Neutrale"
    else:
        return "Negativo"


def get_last_month_http_body():
    ## Ottieni la data di oggi e crea una query che seleziona tutte le richieste di oggi
    query_mese = f"SELECT *FROM whoteach-dev.osquery_dataset.HTTP WHERE DATE(TIMESTAMP_SECONDS(CAST(timestamp AS INT64))) >= DATE_SUB(DATE_TRUNC(CURRENT_DATE(), MONTH), INTERVAL 1 MONTH) AND DATE(TIMESTAMP_SECONDS(CAST(timestamp AS INT64))) < DATE_TRUNC(CURRENT_DATE(), MONTH)"
    # query_settimana = f"SELECT *FROM whoteach-dev.osquery_dataset.HTTP WHERE DATE(TIMESTAMP_SECONDS(CAST(timestamp AS INT64))) >= DATE_SUB(DATE_TRUNC(CURRENT_DATE(), WEEK), INTERVAL 1 WEEK) AND DATE(TIMESTAMP_SECONDS(CAST(timestamp AS INT64))) < DATE_TRUNC(CURRENT_DATE(), WEEK)"

    requests = execute_query_on_bigquery(query_mese)
    ## Filtrami le richieste che hanno body
    requests = requests[requests.astype(str)['body'].map(len) > 0]
    return requests


def get_last_week_http_body():
    ## Ottieni la data di oggi e crea una query che seleziona tutte le richieste di oggi
    # query_mese = f"SELECT *FROM whoteach-dev.osquery_dataset.HTTP WHERE DATE(TIMESTAMP_SECONDS(CAST(timestamp AS INT64))) >= DATE_SUB(DATE_TRUNC(CURRENT_DATE(), MONTH), INTERVAL 1 MONTH) AND DATE(TIMESTAMP_SECONDS(CAST(timestamp AS INT64))) < DATE_TRUNC(CURRENT_DATE(), MONTH)"
    query_settimana = f"SELECT *FROM whoteach-dev.osquery_dataset.HTTP WHERE DATE(TIMESTAMP_SECONDS(CAST(timestamp AS INT64))) >= DATE_SUB(DATE_TRUNC(CURRENT_DATE(), WEEK), INTERVAL 1 WEEK) AND DATE(TIMESTAMP_SECONDS(CAST(timestamp AS INT64))) < DATE_TRUNC(CURRENT_DATE(), WEEK)"
    requests = execute_query_on_bigquery(query_settimana)
    ## Filtrami le richieste che hanno body
    requests = requests[requests.astype(str)['body'].map(len) > 0]
    return requests


def analyze_requests():
    requests = get_last_month_http_body()
    list_of_requests = requests['body'].tolist()
    for request in list_of_requests:

        prompt = 'analyze this http request body: "' + request + '" and evaluate if it could be a cyber threat such as sql injection attack, xss or other types of attacks. Give me a straight "yes", "no" o "maybe" answer and a brief explanation'
        y = model_with_pandas.predict_sqli_pandas(request)
        if y[1] == '1':
            print('sqli pandas')
            print(prompt)
            answer, other_info = evaluate_prompt(prompt)
            print(analizza_sentimento(answer))
            return (
                    'analizzata la richiesta: ' + request + 'per minaccia identificata da modello sqli. Risposta: ' + answer)
        y = model_with_pandas.predict_xss_pandas(request)
        if y[1] == '1':
            print('xss pandas')
            print(prompt)
            answer, other_info = evaluate_prompt(prompt)
            print(analizza_sentimento(answer))
            return (
                    'analizzata la richiesta: ' + request + 'per minaccia identificata da modello sqli. Risposta: ' + answer)

        y = model_with_tf_idf.predict_all_tf_idf(request)
        if y[1] == '1':
            print('all tf_idf')
            print(prompt)
            answer, other_info = evaluate_prompt(prompt)
            print(analizza_sentimento(answer))
            return (
                    'analizzata la richiesta: ' + request + 'per minaccia identificata da modello sqli. Risposta: ' + answer)

        y = model_with_tf_idf.predict_sqli_tf_idf(request)
        if y[1] == '1':
            print('sql_tfidf pandas')
            print(prompt)
            answer, other_info = evaluate_prompt(prompt)
            print(analizza_sentimento(answer))
            return (
                    'analizzata la richiesta: ' + request + 'per minaccia identificata da modello sqli. Risposta: ' + answer)

        y = model_with_tf_idf.predict_xss_tf_idf(request)
        if y[1] == '1':
            print('xss_tfidf pandas')
            print(prompt)
            answer, other_info = evaluate_prompt(prompt)

            print(analizza_sentimento(answer))
            return (
                    'analizzata la richiesta: ' + request + 'per minaccia identificata da modello sqli. Risposta: ' + answer)
    return 'nessuna minaccia trovata'


def analyze_requests_with_llama_last_month():
    requests = get_last_month_http_body()
    list_of_requests = requests['body'].tolist()

    timestamps = requests['timestamp'].tolist()
    # Converti i timestamp in datetime e metti il più recente e il più vecchio in due variabili
    timestamps = [datetime.datetime.fromtimestamp(int(ts)) for ts in timestamps]
    timestamps.sort()
    timestamp_min = timestamps[0]
    timestamp_max = timestamps[-1]
    richieste = ''

    for i, request in enumerate(list_of_requests):
        richieste += '  request ' + str(i) + ': ' + request + '  \n  '

    ## Elimina la cartella /llama/db e tutto ciò che c'è dentro

    try:
        shutil.rmtree(ROOT_DIR + '/llama/db')
    except OSError as e:
        print("Error: %s - %s." % (e.filename, e.strerror))

    try:
        shutil.rmtree(ROOT_DIR + '/llama/source_documents/richieste.txt')
    except OSError as e:
        print("Error: %s - %s." % (e.filename, e.strerror))

    # Assumiamo che timestamp_min sia un oggetto datetime o simile
    nome_mese = calendar.month_name[timestamp_min.month].lower()
    print(nome_mese)

    ## alla prima riga del file /llama/source_documents/richieste.txt scrivi il mese del timestamp minimo
    with open(ROOT_DIR + '/llama/source_documents/richieste.txt', 'w') as f:
        f.write('http requests body of ' + str(nome_mese) + " " + str(
            timestamp_min.year) + ' \n')
    # Scrivi il contenuto di richieste in un file che si chiama richieste.txt
    with open(ROOT_DIR + '/llama/source_documents/richieste.txt', 'a') as f:
        f.write("    request"+str(0)+":  /search?query=' OR '1'='1' --")
        f.write(richieste)

    ingest()
    prompt =  "Can you analyze the http requests body of " + str(nome_mese) +" "+ str(
            timestamp_min.year) + " and tell me if one of them is a cyber attack on my web-app." \
                                  " Look just for attacks based on character sequence lixe xss or sqli, not attacks based on context." \
                                  " If you think you saw a possible attack within a single request, tell me the line of the request."

    answer, other_info = evaluate_prompt(prompt)
    print(answer)
    return answer, timestamp_min, timestamp_max


if __name__ == '__main__':
    analyze_requests_with_llama_last_month()
