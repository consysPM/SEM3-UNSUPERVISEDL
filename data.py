import numpy as np
import pandas as pd
import re
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MultiLabelBinarizer
from collections import Counter



job_keywords = {
    "ITWorker": ["developer", "devops", "sysadmin", "dev", "evangelist"],
    "Designer": ["designer"],
    "Management": ["leadership", "manager", "supervisor", "lead", "executive"],
    "Support": ["support", "advocate"],
    "Sales": ["sales"],
    "Other": []
    }

def normalize_gender(value):
    if pd.isna(value):
        return "Unknown"

    value = str(value).strip().lower()
    tokens = re.findall(r'\b\w+\b', value)  # Wörter extrahieren

    for token in tokens:
        if "female" in token or token == "f":
            return "Female"
        if "male" in token or token == "m":
            return "Male"

    return "Other"

def normalize_age(value):
    if pd.isna(value):
        return np.nan
    try:
        age = int(value)
        if 18 <= age <= 70:
            return age
        else:
            return np.nan
    except ValueError:
        return np.nan

# Funktion: finde Rollen auf Basis sauberer Tokenisierung
def detect_roles(text):
    roles = set()
    if pd.isna(text):
        return roles

    # Alles in lowercase + Tokens extrahieren (Wörter + getrennte Segmente)
    tokens = re.findall(r'\b\w+\b', text.lower())

    for role, keywords in job_keywords.items():
        if any(token in keywords for token in tokens):
            roles.add(role)

    # Wenn keine Rolle gefunden wurde, "Other" hinzufügen
    if not roles:
        roles.add("Other")

    return roles
    
def load_data():
    # Beispielhafte Daten
    df = pd.read_csv('mental-heath-in-tech-2016_20161114.csv')
    #alter normalisieren
    df["Gender_Normalized"] = df["What is your gender?"].apply(normalize_gender)
    # Dummies für Geschlecht erstellen und an DataFrame anhängen
    gender_dummies = pd.get_dummies(df["Gender_Normalized"], prefix="Gender").astype(int)
    df = pd.concat([df, gender_dummies], axis=1)

    #alter normalisieren
    df["Age_Cleaned"] = df["What is your age?"].apply(normalize_age)
    imputer = SimpleImputer(strategy="mean")
    ages_imputed = imputer.fit_transform(df["Age_Cleaned"].values.reshape(-1, 1))
    df["Age_Cleaned"]  = ages_imputed.astype(int)


    # Spalte mit Rollen-Mengen
    df["JobRoles"] = df["Which of the following best describes your work position?"].apply(detect_roles)
    # In DataFrame mit passenden Spaltennamen umwandeln
    mlb = MultiLabelBinarizer()
    roles_encoded = mlb.fit_transform(df["JobRoles"])
    roles_df = pd.DataFrame(roles_encoded, columns=[f"Job_{role}" for role in mlb.classes_])

    # Indizes anpassen & zusammenführen
    roles_df.index = df.index
    df = pd.concat([df, roles_df], axis=1)
    
    #drop
    df.drop(columns=["What is your gender?", "What is your age?"], axis=1, inplace=True)
    df.drop(columns=["Which of the following best describes your work position?", "JobRoles"], inplace=True)
    df.drop(columns=["Gender_Normalized"], inplace=True)


    return df




