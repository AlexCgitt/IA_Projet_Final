import pandas as pd
import numpy as np
import json
import pickle
from sklearn.metrics import r2_score
import argparse  # Importez argparse pour la gestion des arguments de ligne de commande


def estimate_age(json_path, model_user ):
    with open("dict_modeles.pkl", "rb") as f:
        pkl_dict = pickle.load(f)
        print(pkl_dict.keys())
    
    with open(json_path, "r") as f:
        data_json = f.read()


    scaler_X = pkl_dict["scaler_X"]
    scaler_Y = pkl_dict["scaler_Y"]
    model = pkl_dict[model_user]
    encoder_stadedev = pkl_dict["encoder_stadedev"]
    encoder_nomtech = pkl_dict["encoder_nomtech"]

    df = pd.DataFrame(json.loads(data_json))
    X = df[['tronc_diam','haut_tronc', 'haut_tot', 'clc_nbr_diag', 'fk_stadedev', 'fk_nomtech']]
    Y_test = df[['age_estim']]


    # print("O:",ordinal.categories_)
    # print("L:",label.classes_)

    X['fk_stadedev']=pd.DataFrame(encoder_stadedev.transform(X[['fk_stadedev']]))
    X['fk_nomtech'] = pd.DataFrame(encoder_nomtech.transform(X[['fk_nomtech']]))

    X = scaler_X.transform(X)

    pred = model.predict(X) 

    pred = pred.reshape(-1, 1)
    pred = scaler_Y.inverse_transform(pred)

    r2_score_model = r2_score(Y_test, pred)
    print("r2_score = ", r2_score_model)
    
    age_estimated = pd.DataFrame(pred, columns=['age_estim'])
    
    #envoie age_estimated dans un fichier json
    age_estimated.to_json('age_estimated_results.json', orient='records')



parser = argparse.ArgumentParser(description="Estimation de l'âge à partir d'un fichier JSON.")
parser.add_argument('file_path', type=str, help='Chemin vers le fichier JSON contenant les données pour l\'estimation.')
parser.add_argument('--model', type=str, default="RandomForest", help='Modèle à utiliser pour l\'estimation.')

# Analyse des arguments de ligne de commande
args = parser.parse_args()

# Appel de la fonction avec les arguments de ligne de commande
estimate_age(args.file_path, args.model)