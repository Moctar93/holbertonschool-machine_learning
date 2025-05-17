import requests
import pandas as pd
import time

def fetch_data(exchange, filename, limit=2000):
    url = "https://min-api.cryptocompare.com/data/v2/histominute"
    params = {
        "fsym": "BTC",
        "tsym": "USD",
        "e": exchange,
        "limit": limit,  # max 2000 minutes (~33h)
    }

    print(f"Téléchargement de {exchange}...")
    response = requests.get(url, params=params)
    data = response.json()

    if data["Response"] != "Success":
        print("Erreur lors du téléchargement")
        return

    df = pd.DataFrame(data["Data"]["Data"])
    df.rename(columns={
        "time": "Timestamp",
        "volumeto": "Volume_(Currency)",
        "volumefrom": "Volume_(BTC)",
        "open": "Open",
        "high": "High",
        "low": "Low",
        "close": "Close"
    }, inplace=True)

    df["Weighted_Price"] = (df["Volume_(Currency)"] / df["Volume_(BTC)"]).replace([float("inf"), -float("inf")], 0)

    df.to_csv(filename, index=False)
    print(f"{filename} sauvegardé ({len(df)} lignes)")

# Télécharge pour Coinbase et Bitstamp
fetch_data("Coinbase", "coinbase.csv")
time.sleep(1)
fetch_data("Bitstamp", "bitstamp.csv")

