import os

import pandas as pd
from sklearn.model_selection import train_test_split

from business_logic.models_generator import create_model_random_forest

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

if __name__ == '__main__':

    df = pd.read_csv(ROOT_DIR + '/XSS_dataset.csv')

    ## ottieni i nomi delle colonne del dataset
    print(df.columns)


    ## Alle volte funziona se metto 1 come intero e alle volte devo mettere come sotto str(1)
    # ## Copia tutte le colonne df['sentence'] con label 1 in un file chiamato anomalousRequestTestSqli.txt
    df[df['Label'] == 1]['Sentence'].to_csv(ROOT_DIR + '/data/anomalousRequestTestXSS.txt', index=False, header=False)
    ## Copia tutte le colonne df['sentence'] con label 0 in un file chiamato normalRequestTestSqli.txt
    df[df['Label'] == 0]['Sentence'].to_csv(ROOT_DIR + '/data/normalRequestTestXSS.txt', index=False, header=False)


    # ## Copia tutte le colonne df['sentence'] con label 1 in un file chiamato anomalousRequestTestSqli.txt
    # df[df['Label'] == str(1)]['Sentence'].to_csv(ROOT_DIR + '/data/anomalousRequestTestXSS.txt', index=False, header=False)
    # ## Copia tutte le colonne df['sentence'] con label 0 in un file chiamato normalRequestTestSqli.txt
    # df[df['Label'] == str(0)]['Sentence'].to_csv(ROOT_DIR + '/data/normalRequestTestXSS.txt', index=False, header=False)
