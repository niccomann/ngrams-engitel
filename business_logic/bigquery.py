import os
from google.cloud import bigquery
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def execute_query_on_bigquery(query):
    # Imposta le credenziali per l'accesso a BigQuery
    credentials_path = ROOT_DIR + "/service-account.json"  # Sostituisci con il percorso al tuo file di credenziali JSON
    project_id = "whoteach-dev"  # Sostituisci con l'ID del tuo progetto

    # Inizializza il client BigQuery
    client = bigquery.Client.from_service_account_json(credentials_path, project=project_id)
    query_job = client.query(query)
    df = query_job.to_dataframe()
    return df


def get_bodies_from_bigquery():
    table1_name = "osquery_dataset.process_open_sockets"
    table2_name = "osquery_dataset.HTTP"
    table3_name = "osquery_dataset.Utente_IP"

    col1 = "remote_address"
    col2 = "IP"

    query = f"""SELECT *  FROM {table1_name} as t1
        join {table2_name}  as t2 on t1.{col1} = t2.{col2}
    """

    df = execute_query_on_bigquery(query)

    # Crea un df con solo la colonna body
    df_body = df[['body']]

    # Rimuovi tutti gli elementi null da df_body
    df_body = df_body.dropna()

    # Rimuovi gli elementi con lunghezza pari a 0
    df_body = df_body[df_body.astype(str)['body'].map(len) > 0]

    body = df_body['body'].tolist()

    ## da body elimina tutte le stringhe che hanno len < 20
    body = [x for x in body if len(x) > 20]

    return body


if __name__ == '__main__':
    pass

