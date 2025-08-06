import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

def find_optimal_pca_components(df: pd.DataFrame, variance_threshold: float = 0.90):
    """
    Analysiert die optimale Anzahl an Hauptkomponenten mittels der erklärten Varianz.

    Args:
        df (pd.DataFrame): Der sklierte Eingabe-DataFrame.
        variance_threshold (float, optional): Der Schwellenwert für die gewünschte
                                              erklärte Varianz. Standard ist 0.90 (90%).

    Returns:
        int: Die Anzahl der Komponenten, die benötigt werden, um den Schwellenwert zu erreichen.
    """
    # 1. PCA durchführen, um alle Komponenten zu berechnen
    pca = PCA(n_components=None)
    pca.fit(df)

    # 2. Kumulierte erklärte Varianz berechnen
    explained_variance_cumulative = np.cumsum(pca.explained_variance_ratio_)
    
    # 3. Optimale Anzahl an Komponenten finden
    # Wir suchen den ersten Index, an dem der Schwellenwert erreicht oder überschritten wird
    optimal_n_components = np.where(explained_variance_cumulative >= variance_threshold)[0][0] + 1
    
    # 4. Den Scree Plot zeichnen
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(explained_variance_cumulative) + 1), explained_variance_cumulative, marker='o', linestyle='--')
    
    # Hilfslinien und Markierungen hinzufügen
    plt.axhline(y=variance_threshold, color='r', linestyle='--', label=f'{int(variance_threshold*100)}% erklärte Varianz')
    plt.axvline(x=optimal_n_components, color='k', linestyle=':', label=f'Optimale Komponentenanzahl: {optimal_n_components}')
    plt.legend(loc='best')
    
    plt.xlabel('Anzahl der Hauptkomponenten')
    plt.ylabel('Kumulierte erklärte Varianz')
    plt.title('Scree Plot zur Bestimmung der optimalen Komponentenanzahl')
    plt.grid(True)
    plt.show()

    print(f"Um mindestens {int(variance_threshold*100)}% der Varianz zu erklären, werden {optimal_n_components} Komponenten benötigt.")
    
    return optimal_n_components


def evaluate_clusters(df: pd.DataFrame, k: int, pca_components: int = 2):
    df_copy = df.copy()
    # --- 2. Dimensionalitätsreduktion mit PCA für die Visualisierung ---
    pca = PCA(n_components=pca_components)
    # Wende PCA auf die Originaldaten an (ohne die neue 'cluster'-Spalte)
    principal_components = pca.fit_transform(df_copy)
    kmeans = KMeans(n_clusters=k, init='k-means++', random_state=42, n_init=10)
    df_copy['cluster'] = kmeans.fit_predict(principal_components)

   # --- 3. Silhouette Score berechnen ---
    silhouette_avg = silhouette_score(principal_components, df_copy['cluster'])
    print(f'Silhouette Score für k={k}: {silhouette_avg}')

    return silhouette_avg

def visualize_clusters(df: pd.DataFrame, k: int, title: str = 'Cluster-Visualisierung'):
    #Cluster visualisieren ---
    plt.figure(figsize=(12, 8))
    sns.scatterplot(
        x="Hauptkomponente 1",
        y="Hauptkomponente 2",
        hue="cluster",
        palette=sns.color_palette("hsv", n_colors=k),
        data=df,
        legend="full",
        alpha=0.8
    )
    plt.title(f'{title} (k={k})')
    plt.grid(True)
    plt.show()

def reduce_and_cluster(df: pd.DataFrame, k: int, pca_components: int = 2):
    df_copy = df.copy()
    # --- 2. Dimensionalitätsreduktion mit PCA für die Visualisierung ---
    pca = PCA(n_components=pca_components)
    # Wende PCA auf die Originaldaten an (ohne die neue 'cluster'-Spalte)
    principal_components = pca.fit_transform(df_copy)
    kmeans = KMeans(n_clusters=k, init='k-means++', random_state=42, n_init=10)
    df_copy['cluster'] = kmeans.fit_predict(principal_components)

    pca_df = pd.DataFrame(
        data=principal_components, 
        columns=['Hauptkomponente 1', 'Hauptkomponente 2']
    )
    pca_df['cluster'] = df_copy['cluster']

    # --- 3. Cluster visualisieren ---
    # plt.figure(figsize=(12, 8))
    # sns.scatterplot(
    #     x="Hauptkomponente 1",
    #     y="Hauptkomponente 2",
    #     hue="cluster",
    #     palette=sns.color_palette("hsv", n_colors=k),
    #     data=pca_df,
    #     legend="full",
    #     alpha=0.8
    # )
    # plt.title(f'Cluster-Visualisierung mit PCA (k={k})')
    # plt.grid(True)
    # plt.show()

    return pca_df

def cluster_and_reduce(df: pd.DataFrame, k: int):
    """
    Führt K-Means-Clustering und PCA-Visualisierung auf einem DataFrame durch.

    Args:
        df (pd.DataFrame): Der Eingabe-DataFrame mit numerischen Daten.
        k (int): Die gewünschte Anzahl der Cluster.

    Returns:
        pd.DataFrame: Eine Kopie des ursprünglichen DataFrames mit einer
                      zusätzlichen 'cluster'-Spalte.
    """
    # Erstelle eine Kopie, um den ursprünglichen DataFrame nicht zu verändern
    df_copy = df.copy()

    # --- 1. K-Means-Clustering durchführen ---
    kmeans = KMeans(n_clusters=k, init='k-means++', random_state=42, n_init=10)
    df_copy['cluster'] = kmeans.fit_predict(df_copy)

    # --- 2. Dimensionalitätsreduktion mit PCA für die Visualisierung ---
    pca = PCA(n_components=2)
    # Wende PCA auf die Originaldaten an (ohne die neue 'cluster'-Spalte)
    principal_components = pca.fit_transform(df)
    
    pca_df = pd.DataFrame(
        data=principal_components, 
        columns=['Hauptkomponente 1', 'Hauptkomponente 2']
    )
    pca_df['cluster'] = df_copy['cluster']

    # --- 3. Cluster visualisieren ---
    plt.figure(figsize=(12, 8))
    sns.scatterplot(
        x="Hauptkomponente 1",
        y="Hauptkomponente 2",
        hue="cluster",
        palette=sns.color_palette("hsv", n_colors=k),
        data=pca_df,
        legend="full",
        alpha=0.8
    )
    plt.title(f'Cluster-Visualisierung mit PCA (k={k})')
    plt.grid(True)
    plt.show()

    return df_copy

# --- Schritt 1: Daten laden ---
# Annahme: Deine Daten sind in einer Datei namens 'meine_daten.csv' gespeichert.
# Ändere den Dateipfad entsprechend an.
def cluster(df):
    # --- Schritt 2: Optimales k mit der Ellbogenmethode finden ---
    # Wir berechnen die "Within-Cluster Sum of Squares" (WCSS) für verschiedene k-Werte.
    wcss = []
    k_range = range(2, 11)  # Teste k von 1 bis 10

    for k in k_range:
        kmeans = KMeans(n_clusters=k, init='k-means++', random_state=42, n_init=10)
        kmeans.fit(df)
        score = silhouette_score(df, kmeans.labels_)
        print(f'Silhouette Score für k={k}: {score}')
        wcss.append(kmeans.inertia_) # inertia_ ist die WCSS

    # Plot der Ellbogenmethode
    plt.figure(figsize=(10, 6))
    plt.plot(k_range, wcss, marker='o', linestyle='--')
    plt.title('Ellbogenmethode zur Bestimmung des optimalen k')
    plt.xlabel('Anzahl der Cluster (k)')
    plt.ylabel('WCSS (Inertia)')
    plt.xticks(k_range)
    plt.grid(True)
    plt.show()

    # --- Schritt 3: K-Means mit dem optimalen k anwenden ---
    # Wähle das k, an dem der "Ellbogen" im Graphen ist.
    # Beispiel: Wenn der Ellbogen bei k=3 liegt:
    optimal_k = 3 # ÄNDERE DIESEN WERT basierend auf deinem Graphen

    # Erstelle und trainiere das endgültige Modell
    kmeans_final = KMeans(n_clusters=optimal_k, init='k-means++', random_state=42, n_init=10)
    cluster_labels = kmeans_final.fit_predict(df)

    # Füge die Cluster-Zuweisungen zu deinen ursprünglichen Daten hinzu
    df['cluster'] = cluster_labels

    # Zeige die ersten paar Zeilen mit den neuen Cluster-Labels an
    print("\nDaten mit zugewiesenen Clustern:")
    print(df.head())

    # Zeige die Verteilung der Datenpunkte auf die Cluster an
    print("\nVerteilung der Cluster:")
    print(df['cluster'].value_counts())