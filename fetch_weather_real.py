"""
Script para obter dados meteorológicos históricos de Sceaux, França.
Corre este script localmente para obter os dados reais.

Requisitos: pip install requests pandas
"""

import pandas as pd
import requests

# Coordenadas de Sceaux, França (7km de Paris)
LAT = 48.7781
LON = 2.2890

def fetch_weather_data():
    """Obtém dados meteorológicos do Open-Meteo Archive API."""
    
    base_url = "https://archive-api.open-meteo.com/v1/archive"
    
    params = {
        "latitude": LAT,
        "longitude": LON,
        "start_date": "2006-12-01",
        "end_date": "2010-11-30",
        "hourly": ",".join([
            "temperature_2m",
            "relative_humidity_2m", 
            "precipitation",
            "cloud_cover",
            "wind_speed_10m",
            "is_day"
        ]),
        "timezone": "Europe/Paris"
    }
    
    print("A obter dados meteorológicos do Open-Meteo...")
    response = requests.get(base_url, params=params, timeout=120)
    response.raise_for_status()
    data = response.json()
    
    # Converter para DataFrame
    hourly = data["hourly"]
    weather_df = pd.DataFrame({
        "Datetime": pd.to_datetime(hourly["time"]),
        "temperature": hourly["temperature_2m"],
        "humidity": hourly["relative_humidity_2m"],
        "precipitation": hourly["precipitation"],
        "cloud_cover": hourly["cloud_cover"],
        "wind_speed": hourly["wind_speed_10m"],
        "is_day": hourly["is_day"]
    })
    
    # Guardar
    weather_df.to_csv("weather_sceaux_2006_2010.csv", index=False)
    print(f"Dados guardados em weather_sceaux_2006_2010.csv")
    print(f"Shape: {weather_df.shape}")
    print(f"Período: {weather_df['Datetime'].min()} -> {weather_df['Datetime'].max()}")
    
    return weather_df

if __name__ == "__main__":
    df = fetch_weather_data()
    print("\nPrimeiras linhas:")
    print(df.head())
    print("\nEstatísticas:")
    print(df.describe())
