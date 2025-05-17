import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def load_and_merge_data():
    """Charge les fichiers et fusionne les données Coinbase et Bitstamp sur la colonne Timestamp."""
    df_coinbase = pd.read_csv("coinbase.csv")
    df_bitstamp = pd.read_csv("bitstamp.csv")

    # Fusion par Timestamp
    df = pd.merge(df_coinbase, df_bitstamp, on="Timestamp", suffixes=('_cb', '_bs'))

    return df

def preprocess(df):
    """
    Nettoyage et normalisation :
    - Moyenne pondérée entre les deux exchanges
    - Tri temporel
    - Normalisation des valeurs avec MinMaxScaler
    """
    df['vwap'] = df[['Weighted_Price_cb', 'Weighted_Price_bs']].mean(axis=1)
    df = df[['Timestamp', 'vwap']].copy()

    df.dropna(inplace=True)
    df = df.sort_values('Timestamp')

    # Normalisation entre 0 et 1
    scaler = MinMaxScaler()
    df['vwap'] = scaler.fit_transform(df[['vwap']])

    return df, scaler

def create_sequences(df, seq_length=1440):
    """
    Découpe les données en séquences de 1440 minutes (24h) + cible (minute suivante).
    """
    data = df['vwap'].values
    X, y = [], []

    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])    # 24h
        y.append(data[i+seq_length])      # minute suivante

    return np.array(X), np.array(y)

def save_dataset(X, y):
    """Enregistre les données dans un fichier compressé."""
    np.savez_compressed("btc_dataset.npz", X=X, y=y)

if __name__ == "__main__":
    df = load_and_merge_data()
    df, scaler = preprocess(df)
    X, y = create_sequences(df)
    save_dataset(X, y)

