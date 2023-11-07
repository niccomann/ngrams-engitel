import matplotlib.pyplot as plt
import shap


def create_shapely_values_for_one_instance(model, instance_row, df):

    explainer = shap.Explainer(model, check_additivity=False)
    shap_values = explainer.shap_values(instance_row)
    # Ottieni i nomi delle colonne dal DataFrame originale
    feature_names = df.drop('stringa', axis=1).columns.tolist()
    # Converti shap_values in un oggetto Explanation
    explanation = shap.Explanation(shap_values, feature_names=feature_names)
    plt.title('maligno 1, benigno 0:   ' + str(df['label'].iloc[0]))
    shap.plots.bar(explanation)
    # dai come nome al plot la concatenazione di tutte le colonne di instance_row
    nonzero_cols = instance_row[instance_row != 0]
    concatenazione = ' '.join(nonzero_cols.index)
    # plt.title(concatenazione + ' ' + str(df['label'].iloc[0]))
    plt.show()

    # fai il force plot
    shap.plots.force(explanation)


def explain_waterfall(model, df):
    explainer = shap.Explainer(model, df.drop('label', axis=1).drop('stringa', axis=1))
    shap_values = explainer(df.drop('label', axis=1).drop('stringa', axis=1))


    ## Prendimi i valori shap relativi alla prima istanza
    ## (quella che ha label 1, maligno)

    #print('inizio scrittura su csv, json')
    #salva_come_json_e_csv(shap_values, df.drop('label', axis=1).drop('stringa', axis=1).columns) #TODO velocizzare, troppo lento, capire il perchè
    #print('Fine scrittura su csv, json')
    ei = shap_values.values[:, :, 1]
    ## Crea un oggetto Explaination
    idx = 0
    exp = shap.Explanation(shap_values.values[:, :, 1], ## Per tutti gli shap_values dell'output 1
                           shap_values.base_values[:, 1], ## La media degli shap_values dell'output 1
                           data=df.drop('label', axis=1).drop('stringa', axis=1),
                           feature_names=df.drop('label', axis=1).drop('stringa', axis=1).columns)
    plt.title('maligno 1, benigno 0:   ' + str(df['label'].iloc[0]))
    shap.plots.waterfall(exp[idx], show=False)
    plt.savefig('../UI/visualizations/waterfall.png', bbox_inches='tight')

    ##### Pulisci matplotlib che se no mette le figure sovrapposte
    plt.clf()
    plt.cla()
    plt.close()
    shap.plots.bar(exp[idx],show=False)
    plt.savefig('../UI/visualizations/bar.png', bbox_inches='tight')
    ei2 = exp.values[0]
###########
    ##### Questo funziona, ma genera un file html troppo lungo
    shap_plot = shap.force_plot(exp.base_values, exp.values[0], exp.feature_names)
    shap.save_html('../UI/visualizations/force_plot_shap.html', shap_plot)
    ##### Questo funziona, ma genera un file html troppo lungo
###############
