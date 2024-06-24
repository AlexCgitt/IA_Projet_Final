Le fichier script.py contient une fonction qui permet d'estimer l'âge des arbres dont les données sont contenues dans un fichier json, à l'aide de modèles contenus dans dict_modeles.pkl. 
Les différents modèles disponibles sont "RandomForest", "DecisionTree", "MLP", "KNN", "Hist"

Pour le lancer, aller dans le terminal, dans le dossier Besoin_Client_2 et taper dans le terminal : python script.py nom_du_fichier --model nom_du_model 

où nom_du_fichier est le nom du fichier json et nom_du_model est le nom du modèle choisi (cités précédemment)

Par défaut, la ligne sera python script.py Data_Arbre_test.json --model RandomForest
