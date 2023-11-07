import os
import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from business_logic.models_generator import create_model_random_forest

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def loadData(file):
    with open(file, 'r', encoding="utf8") as f:
        data = f.readlines()
    result = []
    for d in data:
        d = d.strip()
        if len(d) > 0:
            result.append(d)
    return result


def create_df_x_y_quante_volte_presente_ngram_in_stringa(bad_requests_path, good_requests_path):
    bad_requests = loadData(bad_requests_path)
    good_requests = loadData(good_requests_path)

    all_requests = bad_requests + good_requests

    yBad = [1] * len(bad_requests)
    yGood = [0] * len(good_requests)
    y = yBad + yGood

    # crea il CountVectorizer per creare gli n-grammi
    vectorizer = CountVectorizer(ngram_range=(3, 3), analyzer='char')

    # crea gli n-grammi per tutte le frasi nella lista all_requests
    X = vectorizer.fit_transform(all_requests)

    # converte la matrice sparsa in un DataFrame e usa i nomi degli n-grammi come nomi di colonne
    df_ngrams = pd.DataFrame(X.toarray(), columns=vectorizer.get_feature_names_out())

    # aggiungi la colonna delle etichette
    df_ngrams['label'] = y

    return df_ngrams, all_requests, vectorizer

def allena_modello_sqli_con_pandas_e_salva():

    print('Inizio df su SQLi')

    df, all_requests, vectorizer = create_df_x_y_quante_volte_presente_ngram_in_stringa(
        ROOT_DIR + '/data/anomalousRequestTestSqli.txt',
        ROOT_DIR + '/data/normalRequestTestSqli.txt')

    # aggiungi a df una colonna con la iesima stringa di all_requests
    df['stringa'] = all_requests

    ## Mischia le righe del df
    df = df.sample(frac=1).reset_index(drop=True)

    # Lascia solo le prime 100 righe del df
    df = df.head(400)

    # dividimi il dataset in train e test
    X_train, X_test, y_train, y_test = train_test_split(df.drop('label', axis=1).drop('stringa', axis=1), df['label'],
                                                        test_size=0.2)

    rfc = create_model_random_forest(X_train, X_test, y_train, y_test)

    joblib.dump(vectorizer, ROOT_DIR + '/models/vectorizer_sqli_pandas.pkl')
    joblib.dump(rfc, ROOT_DIR + '/models/model_sqli_pandas.pkl')

def predict_sqli_pandas(http_request):

    vectorizer = joblib.load(ROOT_DIR + '/models/vectorizer_sqli_pandas.pkl')
    rfc = joblib.load(ROOT_DIR + '/models/model_sqli_pandas.pkl')

    # create_shapely_values_for_one_instance(rfc, X_test.iloc[0], df)
    # explain_waterfall(rfc, df)

    X = vectorizer.transform([http_request])

    df_ngrams = pd.DataFrame(X.toarray(), columns=vectorizer.get_feature_names_out())

    y = rfc.predict(df_ngrams)

    # Trasforma l'ndarray in una stringa
    prediction = np.array2string(y)

    return prediction

def predict_all_pandas(http_request):

    vectorizer = joblib.load(ROOT_DIR + '/models/vectorizer_all_pandas.pkl')
    rfc = joblib.load(ROOT_DIR + '/models/model_all_pandas.pkl')


    X = vectorizer.transform([http_request])

    df_ngrams = pd.DataFrame(X.toarray(), columns=vectorizer.get_feature_names_out())

    y = rfc.predict(df_ngrams)

    # Trasforma l'ndarray in una stringa
    prediction = np.array2string(y)

    return prediction

def allena_modello_xss_con_pandas_e_salva():


    df, all_requests, vectorizer = create_df_x_y_quante_volte_presente_ngram_in_stringa(
        ROOT_DIR + '/data/anomalousRequestAll.txt',
        ROOT_DIR + '/data/normalRequestAll.txt')

    # aggiungi a df una colonna con la iesima stringa di all_requests
    df['stringa'] = all_requests

    ## Mischia le righe del df
    df = df.sample(frac=1).reset_index(drop=True)


    # dividimi il dataset in train e test
    X_train, X_test, y_train, y_test = train_test_split(df.drop('label', axis=1).drop('stringa', axis=1), df['label'],
                                                        test_size=0.2)

    rfc = create_model_random_forest(X_train, X_test, y_train, y_test)


    joblib.dump(vectorizer, ROOT_DIR + '/models/vectorizer_all_pandas.pkl')
    joblib.dump(rfc, ROOT_DIR + '/models/model_all_pandas.pkl')

def predict_xss_pandas(http_request):

    vectorizer = joblib.load(ROOT_DIR + '/models/vectorizer_xss_pandas.pkl')
    rfc = joblib.load(ROOT_DIR + '/models/model_xss_pandas.pkl')

    # create_shapely_values_for_one_instance(rfc, X_test.iloc[0], df)
    # explain_waterfall(rfc, df)

    X = vectorizer.transform([http_request])

    df_ngrams = pd.DataFrame(X.toarray(), columns=vectorizer.get_feature_names_out())

    y = rfc.predict(df_ngrams)

    # Trasforma l'ndarray in una stringa
    prediction = np.array2string(y)

    return prediction


if __name__ == '__main__':
    allena_modello_xss_con_pandas_e_salva()
    y = predict_xss_pandas('//model_sqli_endpoint?corpo_richiesta=s=/Index/\\\\think\\\\app/invokefunction&function=call_user_func_array&vars[0]=md5&vars[1][]=HelloThinkPHP21 HTTP/1.1"')
    print(y)
