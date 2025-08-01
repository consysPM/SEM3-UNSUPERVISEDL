import numpy as np
import pandas as pd
import re
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MultiLabelBinarizer
from collections import Counter
from sklearn.impute import SimpleImputer



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


import pandas as pd

def ordinal_encode_mental_health_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Performs ordinal encoding on specified mental health-related columns in a DataFrame.

    For each column, a custom mapping is applied to transform unique string
    values into numerical ordinal values. New columns are created with
    shorter, logical names ending with '_Score'.

    Args:
        df (pd.DataFrame): The input DataFrame containing the mental health columns.

    Returns:
        pd.DataFrame: A new DataFrame with the specified columns
                      ordinal encoded and renamed.
    """

    df_encoded = df.copy()

    # Define the custom ordinal mappings for each column
    column_mappings = {
        'Does your employer provide mental health benefits as part of healthcare coverage?': {
            'Not eligible for coverage / N/A': 0, 'No': 1, "I don't know": 2, 'Yes': 3
        },
        'Do you know the options for mental health care available under your employer-provided coverage?': {
            'No': 0, 'Unknown': 1, 'I am not sure': 2, 'Yes': 3
        },
        'Has your employer ever formally discussed mental health (for example, as part of a wellness campaign or other official communication)?': {
            'No': 0, "I don't know": 1, 'Yes': 2
        },
        'Does your employer offer resources to learn more about mental health concerns and options for seeking help?': {
            'No': 0, "I don't know": 1, 'Yes': 2
        },
        'Is your anonymity protected if you choose to take advantage of mental health or substance abuse treatment resources provided by your employer?': {
            'No': 0, "I don't know": 1, 'Yes': 2
        },
        'Do you think that discussing a mental health disorder with your employer would have negative consequences?': {
            'Yes': 0, 'Maybe': 1, 'No': 2
        },
        'Do you think that discussing a physical health issue with your employer would have negative consequences?': {
            'Yes': 0, 'Maybe': 1, 'No': 2
        },
        'Would you feel comfortable discussing a mental health disorder with your coworkers?': {
            'No': 0, 'Maybe': 1, 'Yes': 2
        },
        'Would you feel comfortable discussing a mental health disorder with your direct supervisor(s)?': {
            'No': 0, 'Maybe': 1, 'Yes': 2
        },
        'Do you feel that your employer takes mental health as seriously as physical health?': {
            'No': 0, "I don't know": 1, 'Yes': 2
        },
        'Have you heard of or observed negative consequences for co-workers who have been open about mental health issues in your workplace?': {
            'Yes': 0, 'No': 1
        },
        'Have your previous employers provided mental health benefits?': {
            'No, none did': 0, 'Unknown': 1, "I don't know": 2, 'Some did': 3, 'Yes, they all did': 4
        },
        'Were you aware of the options for mental health care provided by your previous employers?': {
            'No, I only became aware later': 0, 'Unknown': 1, 'N/A (not currently aware)': 2,
            'I was aware of some': 3, 'Yes, I was aware of all of them': 4
        },
        'Did your previous employers ever formally discuss mental health (as part of a wellness campaign or other official communication)?': {
            'None did': 0, 'Unknown': 1, "I don't know": 2, 'Some did': 3, 'Yes, they all did': 4
        },
        'Did your previous employers provide resources to learn more about mental health issues and how to seek help?': {
            'None did': 0, 'Unknown': 1, 'Some did': 2, 'Yes, they all did': 3
        },
        'Was your anonymity protected if you chose to take advantage of mental health or substance abuse treatment resources with previous employers?': {
            'No': 0, 'Unknown': 1, "I don't know": 2, 'Sometimes': 3, 'Yes, always': 4
        },
        'Do you think that discussing a mental health disorder with previous employers would have negative consequences?': {
            'Yes, all of them': 0, 'Unknown': 1, "I don't know": 2, 'Some of them': 3, 'None of them': 4
        },
        'Do you think that discussing a physical health issue with previous employers would have negative consequences?': {
            'Yes, all of them': 0, 'Unknown': 1, 'Some of them': 2, 'None of them': 3
        },
        'Would you have been willing to discuss a mental health issue with your previous co-workers?': {
            'No, at none of my previous employers': 0, 'Unknown': 1,
            'Some of my previous employers': 2, 'Yes, at all of my previous employers': 3
        },
        'Would you have been willing to discuss a mental health issue with your direct supervisor(s)?': {
            'No, at none of my previous employers': 0, 'Unknown': 1, "I don't know": 2,
            'Some of my previous employers': 3, 'Yes, at all of my previous employers': 4
        },
        'Did you feel that your previous employers took mental health as seriously as physical health?': {
            'None did': 0, 'Unknown': 1, "I don't know": 2, 'Some did': 3, 'Yes, they all did': 4
        },
        'Did you hear of or observe negative consequences for co-workers with mental health issues in your previous workplaces?': {
            'Yes, all of them': 0, 'Unknown': 1, 'Some of them': 2, 'None of them': 3
        },
        'Would you be willing to bring up a physical health issue with a potential employer in an interview?': {
            'No': 0, 'Maybe': 1, 'Yes': 2
        },
        'Would you bring up a mental health issue with a potential employer in an interview?': {
            'No': 0, 'Maybe': 1, 'Yes': 2
        },
        'Do you feel that being identified as a person with a mental health issue would hurt your career?': {
            'Yes, it has': 0, 'Yes, I think it would': 1, 'Maybe': 2,
            "No, I don't think it would": 3, 'No, it has not': 4
        },
        'Do you think that team members/co-workers would view you more negatively if they knew you suffered from a mental health issue?': {
            'Yes, they do': 0, 'Yes, I think they would': 1, 'Maybe': 2,
            "No, I don't think they would": 3, 'No, they do not': 4
        },
        'How willing would you be to share with friends and family that you have a mental illness?': {
            'Not open at all': 0, 'Somewhat not open': 1, 'Neutral': 2,
            'Not applicable to me (I do not have a mental illness)': 3,
            'Somewhat open': 4, 'Very open': 5
        },
        'Have you observed or experienced an unsupportive or badly handled response to a mental health issue in your current or previous workplace?': {
            'Yes, I experienced': 0, 'Yes, I observed': 1, 'Maybe/Not sure': 2, 'Unknown': 3, 'No': 4
        },
        'Do you have a family history of mental illness?': {
            'No': 0, "I don't know": 1, 'Yes': 2
        },
        'Have you had a mental health disorder in the past?': {
            'No': 0, 'Maybe': 1, 'Yes': 2
        },
        'Do you currently have a mental health disorder?': {
            'No': 0, 'Maybe': 1, 'Yes': 2
        },
        'Have you been diagnosed with a mental health condition by a medical professional?': {
            'No': 0, 'Yes': 1
        },
        'If you have a mental health issue, do you feel that it interferes with your work when being treated effectively?': {
            'Often': 0, 'Sometimes': 1, 'Rarely': 2, 'Never': 3, 'Not applicable to me': 4
        },
        'If you have a mental health issue, do you feel that it interferes with your work when NOT being treated effectively?': {
            'Often': 0, 'Sometimes': 1, 'Rarely': 2, 'Never': 3, 'Not applicable to me': 4
        },
        'Do you work remotely?': {
            'Never': 0, 'Sometimes': 1, 'Always': 2
        }
    }

    # Define a mapping for original column names to shorter names
    column_name_shortenings = {
        'Does your employer provide mental health benefits as part of healthcare coverage?': 'EmployerBenefits',
        'Do you know the options for mental health care available under your employer-provided coverage?': 'KnownOptions',
        'Has your employer ever formally discussed mental health (for example, as part of a wellness campaign or other official communication)?': 'EmployerDiscussed',
        'Does your employer offer resources to learn more about mental health concerns and options for seeking help?': 'EmployerResources',
        'Is your anonymity protected if you choose to take advantage of mental health or substance abuse treatment resources provided by your employer?': 'AnonymityProtected',
        'Do you think that discussing a mental health disorder with your employer would have negative consequences?': 'DiscussMHNegConsequencesEmployer',
        'Do you think that discussing a physical health issue with your employer would have negative consequences?': 'DiscussPHNegConsequencesEmployer',
        'Would you feel comfortable discussing a mental health disorder with your coworkers?': 'ComfortDiscussMHCoWorkers',
        'Would you feel comfortable discussing a mental health disorder with your direct supervisor(s)?': 'ComfortDiscussMHSupervisor',
        'Do you feel that your employer takes mental health as seriously as physical health?': 'EmployerMHPHSeriousness',
        'Have you heard of or observed negative consequences for co-workers who have been open about mental health issues in your workplace?': 'ObservedNegConsequencesCoWorkers',
        'Have your previous employers provided mental health benefits?': 'PrevEmployerBenefits',
        'Were you aware of the options for mental health care provided by your previous employers?': 'PrevEmployerKnownOptions',
        'Did your previous employers ever formally discuss mental health (as part of a wellness campaign or other official communication)?': 'PrevEmployerDiscussed',
        'Did your previous employers provide resources to learn more about mental health issues and how to seek help?': 'PrevEmployerResources',
        'Was your anonymity protected if you chose to take advantage of mental health or substance abuse treatment resources with previous employers?': 'PrevEmployerAnonymityProtected',
        'Do you think that discussing a mental health disorder with previous employers would have negative consequences?': 'DiscussMHNegConsequencesPrevEmployer',
        'Do you think that discussing a physical health issue with previous employers would have negative consequences?': 'DiscussPHNegConsequencesPrevEmployer',
        'Would you have been willing to discuss a mental health issue with your previous co-workers?': 'ComfortDiscussMHPrevCoWorkers',
        'Would you have been willing to discuss a mental health issue with your direct supervisor(s)?': 'ComfortDiscussMHPrevSupervisor',
        'Did you feel that your previous employers took mental health as seriously as physical health?': 'PrevEmployerMHPHSeriousness',
        'Did you hear of or observe negative consequences for co-workers with mental health issues in your previous workplaces?': 'ObservedNegConsequencesPrevCoWorkers',
        'Would you be willing to bring up a physical health issue with a potential employer in an interview?': 'BringUpPHInterview',
        'Would you bring up a mental health issue with a potential employer in an interview?': 'BringUpMHInterview',
        'Do you feel that being identified as a person with a mental health issue would hurt your career?': 'MHIssueHurtCareer',
        'Do you think that team members/co-workers would view you more negatively if they knew you suffered from a mental health issue?': 'CoWorkersNegativeViewMH',
        'How willing would you be to share with friends and family that you have a mental illness?': 'ShareMHFamilyFriends',
        'Have you observed or experienced an unsupportive or badly handled response to a mental health issue in your current or previous workplace?': 'UnsupportiveMHIssueResponse',
        'Do you have a family history of mental illness?': 'FamilyHistoryMH',
        'Have you had a mental health disorder in the past?': 'PastMHDisorder',
        'Do you currently have a mental health disorder?': 'CurrentMHDisorder',
        'Have you been diagnosed with a mental health condition by a medical professional?': 'DiagnosedMHCondition',
        'If you have a mental health issue, do you feel that it interferes with your work when being treated effectively?': 'InterferesWorkTreatedEffectively',
        'If you have a mental health issue, do you feel that it interferes with your work when NOT being treated effectively?': 'InterferesWorkNOTTreatedEffectively',
        'Do you work remotely?': 'WorkRemotely'
    }

    for original_col, mapping in column_mappings.items():
        if original_col in df_encoded.columns:
            new_col_name = f"{column_name_shortenings[original_col]}_Score"
            df_encoded[new_col_name] = df_encoded[original_col].map(mapping)
            df_encoded.drop(columns=[original_col], inplace=True)

        else:
            print(f"Warning: Column '{original_col}' not found in the DataFrame. Skipping.")

    return df_encoded

import pandas as pd
from sklearn.preprocessing import StandardScaler

def scale_dataframe_standard(df: pd.DataFrame) -> pd.DataFrame:
    """
    Skaliert alle numerischen Spalten in einem DataFrame mithilfe des StandardScaler.

    Diese Methode ist für DataFrames gedacht, in denen alle relevanten
    kategorialen Spalten bereits in numerische Form (Ordinal, One-Hot, Multi-Hot)
    kodiert und numerische Spalten (wie Alter) bei Bedarf transformiert wurden.

    Args:
        df (pd.DataFrame): Das DataFrame, dessen Spalten skaliert werden sollen.
                           Alle Spalten sollten numerisch sein (int oder float).

    Returns:
        pd.DataFrame: Ein neues DataFrame mit denselben Spalten, die nun skaliert sind.
                      Die Spaltennamen und der Index bleiben erhalten.
    """
    if not all(pd.api.types.is_numeric_dtype(df[col]) for col in df.columns):
        raise TypeError("Alle Spalten im DataFrame müssen numerisch sein, bevor skaliert wird.")

    scaler = StandardScaler()

    # Den StandardScaler an die Daten anpassen und transformieren
    # Wir übergeben das gesamte DataFrame, damit alle Spalten skaliert werden
    df_scaled_array = scaler.fit_transform(df)

    # Das skalierte NumPy-Array zurück in ein DataFrame konvertieren
    # Wichtig: Spaltennamen und Index beibehalten
    df_scaled = pd.DataFrame(df_scaled_array, columns=df.columns, index=df.index)

    print("DataFrame wurde erfolgreich mit StandardScaler skaliert.")
    print("Beispiel der ersten Zeilen nach Skalierung:")
    print(df_scaled.head())

    return df_scaled

# --- Beispielanwendung (Annahme: df_preprocessed ist Ihr vorbereitetes DataFrame) ---
# Ersetzen Sie dies durch Ihr tatsächliches DataFrame, das alle kodierten Features enthält.
# Beispiel:
# data_example = {
#     'Feature_A_Score': [1, 2, 3, 4, 5],
#     'Feature_B_OneHot_Val1': [0, 1, 0, 1, 0],
#     'Feature_C_OneHot_Val2': [1, 0, 1, 0, 1],
#     'Age_LogTransformed': [3.1, 3.5, 3.7, 3.9, 4.0]
# }
# df_example_preprocessed = pd.DataFrame(data_example)

# df_scaled = scale_dataframe_standard(df_example_preprocessed)

# Hinweis: Wenn Sie Ihre 'ordinal_encode_mental_health_data' Funktion
# und die One-Hot-Encoding-Schritte in einer übergeordneten Funktion
# (wie 'preprocess_for_clustering' aus der vorherigen Antwort) zusammenfassen,
# wäre der Aufruf dieser Skalierungsfunktion der letzte Schritt innerhalb dieser übergeordneten Funktion,
# bevor Sie zur PCA oder direkt zum Clustering übergehen.
    
def load_data():
    # Beispielhafte Daten
    df = pd.read_csv('mental-heath-in-tech-2016_20161114.csv')
    # nur nicht selbständige Personen
    df = df[df["Are you self-employed?"] != 1]
    df.drop(columns=["Are you self-employed?"], inplace=True)

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
    df.drop(columns=["Why or why not?", "Why or why not?.1", "What US state or territory do you work in?", "What US state or territory do you live in?"], inplace=True)


    missing_data = (df.isnull().sum() / df.shape[0]) * 100
    print((missing_data[missing_data >= 50]).sort_values(ascending=False))
    #spalten mit mehr als 50% fehlenden Werten entfernen
    df.dropna(thresh=df.shape[0] * 0.5, axis=1, inplace=True)
    
    #spalten mit fehlenden werten (text) durch unknown ersetzen
    spalten = ["Do you know the options for mental health care available under your employer-provided coverage?",
               "Have your previous employers provided mental health benefits?",
               "Were you aware of the options for mental health care provided by your previous employers?",
               "Did your previous employers ever formally discuss mental health (as part of a wellness campaign or other official communication)?",
               "Did your previous employers provide resources to learn more about mental health issues and how to seek help?",
               "Was your anonymity protected if you chose to take advantage of mental health or substance abuse treatment resources with previous employers?",
            "Do you think that discussing a mental health disorder with previous employers would have negative consequences?",
            "Do you think that discussing a physical health issue with previous employers would have negative consequences?",
            "Would you have been willing to discuss a mental health issue with your previous co-workers?",
            "Would you have been willing to discuss a mental health issue with your direct supervisor(s)?",
            "Did you feel that your previous employers took mental health as seriously as physical health?",
            "Did you hear of or observe negative consequences for co-workers with mental health issues in your previous workplaces?",
            "Have you observed or experienced an unsupportive or badly handled response to a mental health issue in your current or previous workplace?"]

    df[spalten] = df[spalten].fillna("Unknown")

    #Ordinale Werte in numerische Scores umwandeln
    om_leaveEase = {
        "Very easy": 4,
        "Somewhat easy": 3,
        "Neither easy nor difficult": 2,
        "Somewhat difficult": 1,
        "Very difficult": 0,
        "I don't know": np.nan  # wird bewusst auf nan gehalten
    }
    df["LeaveEase_Score"] = df["If a mental health issue prompted you to request a medical leave from work, asking for that leave would be:"].map(om_leaveEase)

    om_orgEmployees = {
        "1-5": 0,
        "6-25": 1,
        "26-100": 2,
        "100-500": 3,
        "500-1000": 4,
        "More than 1000": 5,
    }

    # 2. Imputer definieren – ersetzt NaN durch häufigsten Wert
    imputer = SimpleImputer(strategy="most_frequent")
    df["LeaveEase_Score"] = imputer.fit_transform(df[["LeaveEase_Score"]])
    df["OrgEmployees_Score"] = df["How many employees does your company or organization have?"].map(om_orgEmployees)



    # Spalte entfernen weil mapping stattdessen verwendet wird
    df.drop(columns=["If a mental health issue prompted you to request a medical leave from work, asking for that leave would be:"], inplace=True)
    df.drop(columns=["How many employees does your company or organization have?"], inplace=True)


    # vorerst entfernen (Länder, ordinales mapping nicht sinnvoll, stattdessen evtl. später onehot encoding)
    df.drop(columns=["What country do you live in?", "What country do you work in?"], inplace=True)


    df = ordinal_encode_mental_health_data(df)

    # Log-Transformation für Alter
    df['Age_LogTransformed'] = np.log1p(df['Age_Cleaned'])
    df.drop(columns=['Age_Cleaned'], inplace=True)


    #daten skalieren
    df = scale_dataframe_standard(df)

    #job rollen entfernen (fürs erste)
    #df.drop(columns=[col for col in df.columns if col.startswith('Job_')], inplace=True)

    return df




