import numpy as np
import data
import cluster
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import re
import scipy.stats as stats
from sklearn.metrics import silhouette_score


def main():
    show_cluster_visualization = True
    show_cluster_profiles = True
    #daten vorbereiten
    df = data.load_data()

    #evaluiere optimale anzahl von cluster, beste bewertete cluster-anzahl wird für die weitere analyse verwendet
    hScore = 0
    hK = 0
    for k in range(2, 11):
        score = cluster.evaluate_clusters(df, k=k, pca_components=2)
        if score > hScore:
            hScore = score
            hK = k

    print(f"New best silhouette score: {hScore} for k={hK}")
    pca_df = cluster.reduce_and_cluster(df, k=hK, pca_components=2)

    #visualize the clusters
    if show_cluster_visualization:
        cluster.visualize_clusters(pca_df, k=hK)


    df['cluster'] = pca_df['cluster']


    # --- Schritt 2: Mittelwerte für jeden Cluster berechnen ---
    # Wir gruppieren nach den Clustern und berechnen den Mittelwert für jedes Feature.
    # Die .T (Transpose) am Ende dreht die Tabelle, was den Vergleich oft erleichtert.
    cluster_profiles = df.groupby('cluster').mean().T


    # --- Schritt 3: Profile anzeigen ---
    # Diese Tabelle ist die Grundlage für deine Interpretation.
    print("Cluster-Profile (Mittelwerte der skalierten Merkmale pro Cluster):")
    print(cluster_profiles)

    if show_cluster_profiles:
        # --- Schritt 4: Visualisierung der Cluster-Profile ---
        # Hier verwenden wir eine Heatmap, um die Mittelwerte der Cluster zu visualisieren.
        # Dies hilft, Muster und Unterschiede zwischen den Clustern zu erkennen.
        visualize_clusterprofiles(cluster_profiles)

def visualize_clusterprofiles(cluster_profiles):
    plt.figure(figsize=(12, 10)) # Passe die Größe bei Bedarf an
    sns.heatmap(
        cluster_profiles, 
        annot=True,       # Zeigt die Zahlen in den Zellen an
        cmap="viridis",   # Ein gutes Farbschema
        fmt=".2f"         # Formatiert die Zahlen auf 2 Nachkommastellen
    )
    plt.title('Heatmap der Cluster-Profile')
    plt.show()

    #cluster.cluster(df)
    #cluster.cluster_and_visualize(df, k=2)

    #cluster.cluster_and_plot(df, n_clusters=3)

    # 1. Histogramm und KDE Plot
    # plt.figure(figsize=(12, 5))
    # plt.subplot(1, 2, 1)
    # sns.histplot(df['Age_LogTransformed'], kde=True, bins=20)
    # plt.title(f'Distribution of Age_LogTransformed')
    # plt.xlabel('Age_LogTransformed')
    # plt.ylabel('Frequency')
    # plt.show()

    # Sie können dies für 'If yes, what percentage of your work time...' wiederholen

    #OPTIONAL: BOX-PLOT


if __name__ == "__main__":
    main()