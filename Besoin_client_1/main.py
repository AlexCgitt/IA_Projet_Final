import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from sklearn.metrics import silhouette_score, silhouette_samples, normalized_mutual_info_score



# Chargement du fichier CSV
def charger_data(file):
    return pd.read_csv(file, sep=",", header=0)


# Fonction d'apprentissage des différentes méthodes
def apprentissage(data):
    # Clustering K-Means
    col_data = data[['haut_tot', 'haut_tronc', 'tronc_diam']]
    kmeans = KMeans(n_clusters=5, random_state=42)
    data['cluster'] = kmeans.fit_predict(col_data)

    # Détection d'anomalies Isolation Forest
    ano_data = data[['haut_tronc', 'tronc_diam', 'age_estim', 'fk_prec_estim']]
    iso_forest = IsolationForest(contamination=0.05, random_state=42)
    data['anomaly'] = iso_forest.fit_predict(ano_data)
    data['anomaly'] = data['anomaly'].apply(lambda x: 'Anomalie' if x == -1 else 'Normal')

    # Évaluation du clustering avec plusieurs métriques
    silhouette_avg = silhouette_score(col_data, data['cluster'])
    nmi_avg = normalized_mutual_info_score(data['tronc_diam'], data['cluster'])
    print(f"Indice de silhouette moyen : {silhouette_avg}")
    print(f"Indice de NMI moyen : {nmi_avg}")

    # Déterminer les catégories (petit, moyen, grand)
    stats = data.groupby('cluster')[['haut_tot', 'haut_tronc', 'tronc_diam']].mean().reset_index()
    data['category'] = data.apply(lambda row: categorize(row, stats), axis=1)
    print(stats, "\n", data['category'].value_counts())

    return data, col_data, silhouette_avg


# Interprétation des clusters pour assginer les catégories (petit, moyen, grand)
def categorize(row, stats):
    if row['cluster'] == stats.loc[1, 'cluster']:
        return 'petit'
    elif row['cluster'] == stats.loc[3, 'cluster']:
        return 'petit'
    elif row['cluster'] == stats.loc[0, 'cluster']:
        return 'moyen'
    elif row['cluster'] == stats.loc[2, 'cluster']:
        return 'grand'
    elif row['cluster'] == stats.loc[4, 'cluster']:
        return 'grand'
    else:
        return 'inconnu'


# Déterminer les colonnes numériques pertinentes 
def plot_distribution(data):
    # Plot des colonnes numériques
    num_col = ['haut_tot', 'haut_tronc', 'tronc_diam', 'age_estim', 'fk_prec_estim']
    for col in num_col:
        plt.figure()
        data[col].hist(bins=50)
        plt.title(f'Distribution of {col}')
        plt.xlabel(col)
        plt.ylabel('Frequency')
        plt.show()


# Déterminer le nombre optimal de clusters
def plot_optimal_clusters(col_data):
    distortions = []
    silhouette_scores = []
    K = range(2, 11)
    for k in K:
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(col_data)
        distortions.append(kmeans.inertia_)
        silhouette_scores.append(silhouette_score(col_data, kmeans.labels_))

    # Plot du coefficient de silhouette
    plt.figure(figsize=(10, 6))
    plt.plot(K, silhouette_scores, 'ro-', label='Coefficient de silhouette')
    plt.title('Coefficient de silhouette pour déterminer le nombre optimal de clusters')
    plt.xlabel('Nombre de clusters')
    plt.ylabel('Coefficient de silhouette')
    plt.legend()
    plt.show()

    # Plot de la méthode du coude
    plt.figure(figsize=(10, 6))
    plt.plot(K, distortions, 'bo-', label='Distortion (Inertia)')
    plt.title('Méthode du coude pour déterminer le nombre optimal de clusters')
    plt.xlabel('Nombre de clusters')
    plt.ylabel('Distortion')
    plt.legend()
    plt.show()


# Création du graphique du silhouette d'analyse
def plot_silhouette(col_data, data, silhouette_avg):
    silhouette_vals = silhouette_samples(col_data, data['cluster'])
    fig, ax = plt.subplots(figsize=(10, 6))
    y_upper, y_lower = 0, 0
    for i in range(5):
        cluster_silhouette_vals = silhouette_vals[data['cluster'] == i]
        cluster_silhouette_vals.sort()
        y_upper += len(cluster_silhouette_vals)
        ax.fill_betweenx(np.arange(y_lower, y_upper), 0, cluster_silhouette_vals, alpha=0.7)
        ax.text(-0.05, y_lower + 0.5 * len(cluster_silhouette_vals), str(i + 1))
        y_lower += len(cluster_silhouette_vals)
    ax.axvline(x=silhouette_avg, color="red", linestyle="--")
    ax.set_title("Graphique de silhouette des clusters")
    ax.set_xlabel("Coefficient de silhouette")
    ax.set_ylabel("Cluster")
    ax.set_yticks([])
    ax.set_xticks([-0.1, 0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])
    plt.show()

                          
# Création de la carte avec plotly et mise en forme avec mapbox en fonction de la catégorie choisie
def carte(data, cat):
    if cat in ['petit', 'moyen', 'grand']:
        cat_data = data[data['category'] == cat]
        fig = px.scatter_mapbox(cat_data, lat="latitude", lon="longitude", color="category", zoom=12)
        fig.update_layout(mapbox_style="open-street-map")
        fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
        fig.show()
    elif cat == 'tout':
        fig = px.scatter_mapbox(data, lat="latitude", lon="longitude", color="category", zoom=12)
        fig.update_layout(mapbox_style="open-street-map")
        fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
        fig.show()
    elif cat == 'anomalie':
        fig = px.scatter_mapbox(data, lat="latitude", lon="longitude", color="anomaly", zoom=12)
        fig.update_layout(mapbox_style="open-street-map")
        fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
        fig.show()
    else:
        print("Catégorie invalide.")


# Script qui execute le code
def main():
    file = input("Veuillez choisir un fichier CSV : ")
    data = charger_data(file)
    data, col_data, silhouette_avg = apprentissage(data)
    while True:
        category = input("Veuillez choisir une catégorie (petit, moyen, grand, tout, anomalie, plots, silhouette, clusters) ou 'q' pour quitter : ")
        if category.lower() == 'q':
            break
        elif category.lower() == 'plots':
            plot_distribution(data)
        elif category.lower() == 'clusters':
            plot_optimal_clusters(col_data)
        elif category.lower() == 'silhouette':
            plot_silhouette(col_data, data, silhouette_avg)
        else:
            carte(data, category)



if __name__ == "__main__":
    main()