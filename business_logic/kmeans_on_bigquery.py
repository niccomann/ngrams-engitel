from business_logic.bigquery import execute_query_on_bigquery


def ottieni_table_generata_com_kmeans():
    query = "SELECT * FROM ML.DETECT_ANOMALIES(MODEL `whoteach-dev.osquery_dataset.my_kmeans_model`,  STRUCT(0.02 AS contamination), TABLE `whoteach-dev.osquery_dataset.HTTP`)"
    df = execute_query_on_bigquery(query)
    # Ritorna solo le prime 5 righe della tabella
    df = df.head(5)
    return df

if __name__ == '__main__':

    print('inizio kmeans_on_bigquery')
    #query = "CREATE MODEL `whoteach-dev.osquery_dataset.my_kmeans_model` OPTIONS( MODEL_TYPE = 'kmeans', NUM_CLUSTERS = 8, KMEANS_INIT_METHOD = 'kmeans++') AS SELECT * EXCEPT(timestamp) FROM `whoteach-dev.osquery_dataset.HTTP`;"
    #df = execute_query_on_bigquery(query)
    query = "SELECT * FROM ML.DETECT_ANOMALIES(MODEL `whoteach-dev.osquery_dataset.my_kmeans_model`,  STRUCT(0.02 AS contamination), TABLE `whoteach-dev.osquery_dataset.HTTP`)"
    df = execute_query_on_bigquery(query)
    print('fine kmeans_on_bigquery')
    print(df)
