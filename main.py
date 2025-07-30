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
    df = data.load_data()
    #df.info()
    #df.describe(include='all')
    
    df.to_clipboard(index=False)
    

    cluster.cluster(df)
    cluster.cluster_and_visualize(df, k=2)

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

if __name__ == "__main__":
    main()