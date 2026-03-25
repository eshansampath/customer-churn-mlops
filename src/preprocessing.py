import pandas as pd

def clean_data(df):
    df = df.copy()

    # Fix TotalCharges
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

    # Drop missing rows
    df = df.dropna()

    # Clean Churn column
    df['Churn'] = df['Churn'].astype(str).str.strip()

    # Map values
    df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})

    # Drop invalid rows
    df = df.dropna(subset=['Churn'])

    # Convert to int
    df['Churn'] = df['Churn'].astype(int)

    return df