import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import tree
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from business_logic.model_with_tf_idf import create_vectorized_x_y

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def create_model_logistic_regression(X_train, X_test, y_train, y_test):
    ## Stampami i valori di X e Y
    print("X_train: ", X_train.shape, " y_train: ", len(y_train))
    print("X_test: ", X_test.shape, " y_test: ", len(y_test))
    # logistic regression
    lgs = LogisticRegression()
    lgs.fit(X_train, y_train)
    y_pred = lgs.predict(X_test)
    score_test = metrics.accuracy_score(y_test, y_pred)
    print("Score Logistic Regression :", score_test)

    return lgs


def create_model_decison_tree(X_train, X_test, y_train, y_test):
    # Decesion Tree
    dtc = tree.DecisionTreeClassifier()
    dtc.fit(X_train, y_train)
    y_pred = dtc.predict(X_test)
    score_test = metrics.accuracy_score(y_test, y_pred)
    print("Score Decesion Tree :", score_test)
    return dtc


def create_model_random_forest(X_train, X_test, y_train, y_test):
    # Random Forest
    rfc = RandomForestClassifier(n_estimators=200)
    rfc.fit(X_train, y_train)
    y_pred = rfc.predict(X_test)
    score_test = metrics.accuracy_score(y_test, y_pred)
    print("Score Random Forest :", score_test)

    return rfc




if __name__ == '__main__':
    print('Inizio tfIdf su SQLi')

    X, y, vectorizer = create_vectorized_x_y('./data/anomalousRequestTestSqli.txt', './data/normalRequestTestSqli.txt')

    X_train_Tfidf, X_test_Tfidf, y_train, y_test = train_test_split(X, y, test_size=0.002, random_state=21)

    rfc = create_model_random_forest(X_train_Tfidf, X_test_Tfidf, y_train, y_test)

    print('Inizio tfIdf CISC')

    X, y, vectorizer = create_vectorized_x_y('./data/anomalousRequestAll.txt', './data/normalRequestAll.txt')

    X_train_Tfidf, X_test_Tfidf, y_train, y_test = train_test_split(X, y, test_size=0.002, random_state=21)

    rfc = create_model_random_forest(X_train_Tfidf, X_test_Tfidf, y_train, y_test)

    matrice_sparsa = X_test_Tfidf[0]

    # Sostituisci i NaN con zero
    dense_matrix = np.nan_to_num(matrice_sparsa)

    ## TODO per verificare che effettivamente la explainabiluty sia giusta è possibile vedere il modello addestrato con df e con tfidf e vedere se gli shap risultano uguali, simili
