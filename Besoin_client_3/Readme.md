# Client 3: Système d'Alerte pour les Tempêtes

##### Objectif: Mettre en place un système d'alerte pour prédire la position des arbres susceptibles d'être déracinés en cas de tempête.

## Importation des Librairies

Dans cette section, nous importons les librairies nécessaires pour notre analyse. Les librairies incluent `pandas`, `scipy.stats`, `seaborn`, `matplotlib.pyplot`, `numpy`, `sklearn`, `imblearn`, et `joblib`.

**Où lancer:** Dans la cellule Python après le titre "Importation des librairies".

## Importation de la Base de Données

Nous chargeons le dataset "Data_Arbre.csv" dans un DataFrame et affichons les premières lignes pour une inspection initiale.

**Où lancer:** Dans la cellule Python après le titre "importation de la base de donnée".

## Etape 1: Préparation des Données

### 1. Extraction des Données d’Intérêt

Sélection des colonnes pertinentes du dataset pour prédire la cible `fk_arb_etat`.

**Où lancer:** Dans la cellule Markdown après le sous-titre "Extraction des données d’intérêt".

### 2. Visualisation des Données

Nous utilisons des histogrammes pour observer la distribution des différentes variables et leur corrélation avec la variable cible.

**Où lancer:** Dans les cellules Python suivant les sous-titres décrivant les manières différentes de faire (histplots).

### 3. Test de Chi2 pour les Variables Qualitatives

Utilisation du test de chi2 pour déterminer si les variables qualitatives sont dépendantes de l'état de l'arbre.

**Où lancer:** Dans les cellules Python après le sous-titre "Test de chi2 pour les variables qualitatives".

### 4. Encodage et Normalisation des Données

Encodage des variables catégorielles et standardisation des données pour les mettre à la même échelle.

**Où lancer:** Dans les cellules Python après le sous-titre "Encodage des données catégorielles et normalisation des données".

## Etape 2: Apprentissage Supervisé pour la Classification

### 1. Séparation des Données

Division des données en ensemble d'entraînement et ensemble de test (70% pour l'entraînement et 30% pour le test).

**Où lancer:** Dans la cellule Python après le sous-titre "Séparation des données".

### 2. Utilisation de SMOTE

Application de SMOTE pour traiter le déséquilibre des classes dans l'ensemble d'entraînement.

**Où lancer:** Dans la cellule Python après le sous-titre "Utilisation d'un SMOTE sur l'ensemble d'entraînement".

### 3. Choix de l'Algorithme d'Apprentissage

Utilisation de `RandomForestClassifier` avec `GridSearchCV` pour optimiser les hyperparamètres du modèle.

**Où lancer:** Dans les cellules Python après le sous-titre "Choix de l'algorithme d'apprentissage" et "Utilisation du GridSearchCV".

### 4. Sélection des Caractéristiques Importantes

Affichage des caractéristiques les plus importantes après optimisation des hyperparamètres.

**Où lancer:** Dans la cellule Python après le sous-titre "feature selection".

### 5. Entraînement du Modèle

Entraînement du modèle avec les caractéristiques les plus importantes et prédiction des résultats sur l'ensemble de test.

**Où lancer:** Dans la cellule Python après le sous-titre "Entrainement de notre modèle d'apprentissage".

## Etape 3: Métriques pour la Classification

### 1. Evaluation des Résultats de la Classification

Evaluation du modèle avec des métriques telles que l'accuracy, le rapport de classification et la matrice de confusion.

**Où lancer:** Dans la cellule Python après le sous-titre "Evaluation de résultat de la classification".

### 2. Résultats

Le modèle a montré une haute précision pour la classe majoritaire mais a eu des performances limitées pour la classe minoritaire.

## Etape 4: Préparation du Script

Pour exécuter le modèle, utilisez le script fourni dans le fichier `script2.py`. Dans le terminal, lancez `python script2.py`.

## Etape 5: Fonctionnalité Bonus

Des fonctionnalités supplémentaires peuvent être ajoutées ici.