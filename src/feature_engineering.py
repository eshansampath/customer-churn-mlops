import pandas as pd

def create_features(df):
    df['CLV'] = df['MonthlyCharges'] * df['tenure']

    df['tenure_group'] = pd.cut(
        df['tenure'],
        bins=[0, 12, 24, 48, 60, 100],
        labels=["0-1yr", "1-2yr", "2-4yr", "4-5yr", "5+yr"]
    )

    return df