import data
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


def main():
    df = data.load_data()

    df = df.dropna(subset=["Is your primary role within your company related to tech/IT?"])

    df.to_clipboard(index=False)  # Kopiere DataFrame in die Zwischenablage

    corr = df["Job_ITWorker"].corr(df["Is your primary role within your company related to tech/IT?"])
    print(f"Korrelation: {corr:.3f}")

    #print(df["Is your primary role within your company related to tech/IT?"].head(100))

    df["Job_TechRole"] = ((df["Job_ITWorker"] == 1)).astype(int)
    
    # Annahme: Spalte enthält "Yes"/"No" oder 1/0
    df_clean = df[df["Is your primary role within your company related to tech/IT?"].isin([1, 0])]


    # Kreuztabelle erzeugen
    ct = pd.crosstab(
        df_clean["Job_TechRole"],
        df_clean["Is your primary role within your company related to tech/IT?"],
        normalize="index"  # für Prozent je Zeile (optional)
    )

    

    # Heatmap anzeigen
    plt.figure(figsize=(6, 4))
    sns.heatmap(ct, annot=True, cmap="Blues", fmt=".1%", cbar=False)
    plt.title("Wahrnehmung vs. Erkannt (Job_TechRole)")
    plt.xlabel("Selbsteinschätzung: Tech/IT?")
    plt.ylabel("System-Kategorie: IT-Rolle erkannt?")
    plt.yticks([0.5, 1.5], ["Nein", "Ja"], rotation=0)
    plt.tight_layout()
    plt.show()



if __name__ == "__main__":
    main()