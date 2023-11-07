import os
import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from business_logic import models_generator
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def loadData(file):
    with open(file, 'r', encoding="utf8") as f:
        data = f.readlines()
    result = []
    for d in data:
        d = d.strip()
        if (len(d) > 0):
            result.append(d)
    return result

def create_vectorized_x_y(bad_requests_path, good_requests_path):
    bad_requests = loadData(bad_requests_path)
    good_requests = loadData(good_requests_path)

    all_requests = bad_requests + good_requests

    yBad = [1] * len(bad_requests)
    yGood = [0] * len(good_requests)
    y = yBad + yGood

    vectorizer = TfidfVectorizer(min_df=0.0, analyzer="char", sublinear_tf=True, ngram_range=(3, 3))
    X = vectorizer.fit_transform(all_requests)

    return X, y, vectorizer



def allena_modello_sqli_con_tfidf_e_salva():

    X, y, vectorizer = models_generator.create_vectorized_x_y(ROOT_DIR+'/data/anomalousRequestTestSqli.txt',
                                                              ROOT_DIR+'/data/normalRequestTestSqli.txt')

    X_train_Tfidf, X_test_Tfidf, y_train, y_test = train_test_split(X, y, test_size=0.002, random_state=21)


    rfc = models_generator.create_model_random_forest(X_train_Tfidf, X_test_Tfidf, y_train, y_test)

    joblib.dump(vectorizer, ROOT_DIR + '/models/vectorizer_sqli_tfidf.pkl')

    joblib.dump(rfc, ROOT_DIR + '/models/model_sqli_tfidf.pkl')




def predict_sqli_tf_idf(http_request):

    vectorizer = joblib.load(ROOT_DIR + '/models/vectorizer_sqli_tfidf.pkl')
    rfc = joblib.load(ROOT_DIR + '/models/model_sqli_tfidf.pkl')

    ## Con la funzione trasform, x viene vettorizzato con il vectorizer già addestrato e restituisce un vettore con la dimensione giusta
    ## (attenzione a non utilizzare fit_trasform che invece non utilizza un vectorizer addestrato )
    x = vectorizer.transform([http_request])

    y_pred = rfc.predict(x)
    print(y_pred)

    # trasforma l'ndarray in una stringa
    y_pred = np.array2string(y_pred)

    return y_pred



def predict_all_tf_idf(http_request):

    vectorizer = joblib.load(ROOT_DIR + '/models/vectorizer_all_tfidf.pkl')
    rfc = joblib.load(ROOT_DIR + '/models/model_all_tfidf.pkl')
    ## Con la funzione trasform, x viene vettorizzato con il vectorizer già addestrato e restituisce un vettore con la dimensione giusta
    ## (attenzione a non utilizzare fit_trasform che invece non utilizza un vectorizer addestrato )
    x = vectorizer.transform([http_request])
    y_pred = rfc.predict(x)
    print(y_pred)
    # trasforma l'ndarray in una stringa
    y_pred = np.array2string(y_pred)

    return y_pred


def allena_modello_xss_con_tfidf_e_salva():
    X, y, vectorizer = models_generator.create_vectorized_x_y(ROOT_DIR + '/data/anomalousRequestTestXSS.txt',
                                                              ROOT_DIR + '/data/normalRequestTestXSS.txt')

    X_train_Tfidf, X_test_Tfidf, y_train, y_test = train_test_split(X, y, test_size=0.002, random_state=21)

    rfc = models_generator.create_model_random_forest(X_train_Tfidf, X_test_Tfidf, y_train, y_test)

    joblib.dump(vectorizer, ROOT_DIR + '/models/vectorizer_xss_tfidf.pkl')
    joblib.dump(rfc, ROOT_DIR + '/models/model_xss_tfidf.pkl')


def predict_xss_tf_idf(http_request):

    rfc = joblib.load(ROOT_DIR + '/models/model_xss_tfidf.pkl')
    vectorizer = joblib.load(ROOT_DIR + '/models/vectorizer_xss_tfidf.pkl')

    ## Con la funzione trasform, x viene vettorizzato con il vectorizer già addestrato e restituisce un vettore con la dimensione giusta
    ## (attenzione a non utilizzare fit_trasform che invece non utilizza un vectorizer addestrato )
    x = vectorizer.transform([http_request])


    y_pred = rfc.predict(x)

    print(y_pred)

    # trasforma l'ndarray in una stringa
    y_pred = np.array2string(y_pred)

    return y_pred

if __name__ == '__main__':
    predict_sqli_tf_idf('GET /search?query=1%27%20or%20%271%27=%271 HTTP/1.1')
    print('-------------------')
    predict_xss_tf_idf('GET /search?query=<script>alert('')</script> HTTP/1.1')
    print('-------------------')
