"""
=============================================================================
📋 PIPELINE FINAL RECOMENDADA - PREVISÃO DE CONSUMO MENSAL
=============================================================================

AUTOR: Álvaro Terroso
CURSO: Machine Learning Algorithms - MAAI - IPCA
DATASET: Individual Household Electric Power Consumption (UCI)

=============================================================================
🎯 OBJETIVO
=============================================================================
Prever o consumo energético total do próximo mês para uma habitação,
usando dados históricos de consumo e meteorologia.

=============================================================================
📊 RESULTADOS COM DATASET DE AMOSTRA (10K registos, ~14 meses)
=============================================================================

| Estratégia                         | Erro Médio Mensal |
|------------------------------------|-------------------|
| Persistência Semanal (lag_168)     | 7.38%            |  ← MELHOR para dados limitados
| Modelo Direto (Cal + Meteo)        | 17.01%           |
| Baseline + Ajuste                  | 26.36%           |
| Modelo Recursivo (com lags)        | 65.97%           |  ← Pior (acumula erro)

=============================================================================
🏆 ESTRATÉGIA RECOMENDADA
=============================================================================

PARA DATASET LIMITADO (<2 anos):
    → Usar PERSISTÊNCIA SEMANAL (lag_168)
    → Simples, robusto, sem overfitting
    → Erro esperado: 5-10%

PARA DATASET COMPLETO (>2 anos com ciclos sazonais):
    → Usar ENSEMBLE: Persistência + Modelo de Ajuste
    → O modelo aprende correções sazonais com dados de anos anteriores
    → Erro esperado: 3-7%

=============================================================================
💡 LIÇÕES APRENDIDAS
=============================================================================

1. PREVISÃO RECURSIVA é perigosa para horizontes longos
   - Erro acumula-se exponencialmente
   - Evitar para previsões >24h sem re-ancoragem

2. PERSISTÊNCIA é um baseline muito forte em séries de consumo
   - Comportamento humano é repetitivo
   - Difícil de bater sem dados sazonais completos

3. METEOROLOGIA ajuda, mas requer calibração sazonal
   - Precisa de pelo menos 1 ciclo completo (12 meses) de treino
   - Idealmente 2+ anos para capturar variabilidade

4. FEATURES CÍCLICAS são essenciais
   - sin/cos para hora, dia da semana, mês
   - Capturam periodicidade sem descontinuidades

=============================================================================
"""

import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

def load_data(consumption_path, weather_path):
    """Carrega e junta dados de consumo e meteorologia."""
    df = pd.read_csv(consumption_path, parse_dates=["Datetime"])
    weather = pd.read_csv(weather_path, parse_dates=["Datetime"])
    df = df.merge(weather, on="Datetime", how="left")
    df = df.set_index("Datetime").sort_index()
    return df

def create_features(df, target_col="target_kwh_hour"):
    """Cria features necessárias."""
    df = df.copy()
    
    # Calendário
    df["hour"] = df.index.hour
    df["day_of_week"] = df.index.dayofweek
    df["month"] = df.index.month
    
    # Encoding cíclico
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["dow_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    
    # Meteorologia derivada
    if "temperature" in df.columns:
        df["heating_degree"] = np.maximum(0, 18 - df["temperature"])
        df["cooling_degree"] = np.maximum(0, df["temperature"] - 24)
    
    # Lags (para persistência e modelo com lags)
    df["lag_1"] = df[target_col].shift(1)
    df["lag_24"] = df[target_col].shift(24)
    df["lag_168"] = df[target_col].shift(168)
    
    return df.dropna()

def forecast_persistence(df, month_start, month_end, target_col="target_kwh_hour"):
    """
    Previsão por PERSISTÊNCIA SEMANAL.
    Usa o valor da mesma hora há 1 semana (lag_168).
    """
    forecast_range = pd.date_range(month_start, month_end, freq="h")
    predictions = []
    
    for ts in forecast_range:
        lag_ts = ts - pd.Timedelta(hours=168)
        if lag_ts in df.index:
            pred = df.loc[lag_ts, target_col]
        else:
            pred = df[target_col].mean()
        predictions.append(max(0, pred))
    
    return pd.Series(predictions, index=forecast_range)

def evaluate_month(y_true, y_pred):
    """Calcula métricas de avaliação."""
    common = y_true.index.intersection(y_pred.index)
    if len(common) == 0:
        return None
    
    y_true = y_true.loc[common]
    y_pred = y_pred.loc[common]
    
    total_true = y_true.sum()
    total_pred = y_pred.sum()
    pct_error = abs(total_pred - total_true) / total_true * 100
    mae = mean_absolute_error(y_true, y_pred)
    
    return {
        "n_hours": len(common),
        "total_true_kwh": round(total_true, 2),
        "total_pred_kwh": round(total_pred, 2),
        "abs_error_kwh": round(abs(total_pred - total_true), 2),
        "pct_error": round(pct_error, 2),
        "mae_hourly": round(mae, 4)
    }

def walk_forward_persistence(df, first_month, n_months, target_col="target_kwh_hour"):
    """
    Walk-forward validation com persistência semanal.
    """
    results = []
    first_month_start = pd.Timestamp(first_month + "-01")
    
    for i in range(n_months):
        month_start = first_month_start + pd.DateOffset(months=i)
        month_end = (month_start + pd.offsets.MonthEnd(1)).normalize() + pd.Timedelta(hours=23)
        month_str = month_start.strftime("%Y-%m")
        
        if month_end > df.index.max():
            break
        
        y_pred = forecast_persistence(df, month_start, month_end, target_col)
        y_true = df.loc[month_start:month_end, target_col]
        
        metrics = evaluate_month(y_true, y_pred)
        if metrics:
            metrics["month"] = month_str
            results.append(metrics)
    
    return pd.DataFrame(results)

# =============================================================================
# DEMONSTRAÇÃO
# =============================================================================
if __name__ == "__main__":
    print("="*70)
    print("🔌 PIPELINE DE PREVISÃO - DEMONSTRAÇÃO")
    print("="*70)
    
    # Carregar
    df = load_data(
        "hourly.csv",
        "weather_sceaux_2006_2010.csv"
    )
    
    # Features
    df = create_features(df)
    print(f"\n📊 Dados: {len(df)} registos")
    print(f"📅 Período: {df.index.min()} -> {df.index.max()}")
    
    # Walk-forward
    print("\n📈 Resultados - Persistência Semanal:")
    results = walk_forward_persistence(df, "2010-1", 3)
    print(results[["month", "total_true_kwh", "total_pred_kwh", "pct_error"]].to_string(index=False))
    
    print(f"\n🎯 ERRO MÉDIO MENSAL: {results['pct_error'].mean():.2f}%")
    print(f"📊 MAE MÉDIO: {results['mae_hourly'].mean():.4f} kWh")
