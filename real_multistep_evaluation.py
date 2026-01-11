"""
=============================================================================
AVALIAÇÃO MULTI-STEP REAL - PREVISÃO DE MÊS INTEIRO "A SECO"
=============================================================================

CENÁRIO DE PRODUÇÃO REAL:
- Estamos no dia 31 de Janeiro
- Queremos prever TODO o mês de Fevereiro
- SÓ temos informação histórica até 31 de Janeiro
- NÃO temos lags de 1h, 24h durante Fevereiro (ainda não aconteceram!)

O QUE PODEMOS USAR:
- Lags >= 1 mês (lag_720+ horas) - do mês anterior
- Features de calendário (hora, dia da semana, mês)
- Meteorologia (previsão ou histórica do ano anterior)
- Padrões sazonais aprendidos

O QUE NÃO PODEMOS USAR:
- lag_1 (hora anterior) - não existe!
- lag_24 (dia anterior) - não existe!
- lag_168 (semana anterior) - só existe para última semana do mês!

=============================================================================
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


def get_season(month):
    if month in [12, 1, 2]:
        return "Inverno"
    elif month in [3, 4, 5]:
        return "Primavera"
    elif month in [6, 7, 8]:
        return "Verão"
    else:
        return "Outono"


def create_features_production(df, target_col, forecast_start):
    """
    Cria features para PRODUÇÃO REAL.
    
    Para previsão do mês M, só podemos usar:
    - Lags >= 720h (1 mês) - calculados com dados até forecast_start
    - Features de calendário (conhecidas antecipadamente)
    - Meteorologia (previsão ou média histórica)
    
    NÃO podemos usar:
    - lag_1, lag_24, lag_168 durante o mês de previsão
    """
    df = df.copy()
    
    # =========================================================================
    # FEATURES DE CALENDÁRIO (sempre disponíveis)
    # =========================================================================
    df["hour"] = df.index.hour
    df["day_of_week"] = df.index.dayofweek
    df["day_of_month"] = df.index.day
    df["month"] = df.index.month
    df["is_weekend"] = (df.index.dayofweek >= 5).astype(int)
    
    # Encoding cíclico
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["dow_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    
    # =========================================================================
    # LAGS DE LONGO PRAZO (disponíveis em produção)
    # =========================================================================
    
    # Lag de 1 mês (~720 horas)
    df["lag_720"] = df[target_col].shift(720)
    
    # Lag de 1 ano (8760 horas) - mesma época do ano passado
    df["lag_8760"] = df[target_col].shift(8760)
    
    # Média do mesmo mês/hora do histórico (perfil típico)
    # Isto é calculado só com dados históricos
    
    # =========================================================================
    # FEATURES DE PERFIL HISTÓRICO
    # =========================================================================
    
    # Média por hora do dia (padrão diário)
    hourly_profile = df.loc[:forecast_start, target_col].groupby(
        df.loc[:forecast_start].index.hour
    ).mean()
    df["hourly_profile"] = df.index.hour.map(hourly_profile)
    
    # Média por hora + dia da semana (padrão semanal)
    weekly_profile = df.loc[:forecast_start, target_col].groupby(
        [df.loc[:forecast_start].index.dayofweek, df.loc[:forecast_start].index.hour]
    ).mean()
    df["weekly_profile"] = df.apply(
        lambda x: weekly_profile.get((x.name.dayofweek, x.name.hour), np.nan), 
        axis=1
    )
    
    # Média por mês + hora (padrão sazonal)
    monthly_profile = df.loc[:forecast_start, target_col].groupby(
        [df.loc[:forecast_start].index.month, df.loc[:forecast_start].index.hour]
    ).mean()
    df["monthly_profile"] = df.apply(
        lambda x: monthly_profile.get((x.name.month, x.name.hour), np.nan),
        axis=1
    )
    
    return df

def forecast_month_recursive(model, df, train_end, forecast_start, forecast_end):
    """
    Previsão REAL recursiva - como seria em produção.
    """
    # Histórico disponível
    history = df[df.index <= train_end].copy()
    
    predictions = []
    
    # Gerar horas a prever
    forecast_hours = pd.date_range(forecast_start, forecast_end, freq='H')
    
    for hour in forecast_hours:
        # Criar features para esta hora
        features = create_features_for_hour(hour, history)
        
        # Prever
        pred = model.predict([features])[0]
        predictions.append(pred)
        
        # ATUALIZAR histórico com a previsão (não com valor real!)
        history.loc[hour, target_col] = pred  # ← PREVISÃO, não real!
    
    return predictions


def forecast_month_real(df, target_col, train_end, month_to_forecast, include_meteo=True):
    """
    Previsão REAL de um mês inteiro.
    
    Simula produção: treina até train_end, prevê mês seguinte sem lags curtos.
    """
    
    # Definir período de previsão
    forecast_start = month_to_forecast
    forecast_end = forecast_start + pd.DateOffset(months=1) - pd.Timedelta(hours=1)
    
    # Criar features
    df_feat = create_features_production(df, target_col, train_end)
    
    # Features disponíveis em produção
    feature_cols = [
        # Calendário
        "is_weekend", "hour_sin", "hour_cos", "dow_sin", "dow_cos",
        "month_sin", "month_cos", "day_of_month",
        # Lags longos
        "lag_720",   # 1 mês atrás
        "lag_8760",  # 1 ano atrás
        # Perfis históricos
        "hourly_profile", "weekly_profile", "monthly_profile"
    ]
    
    # Adicionar meteorologia se disponível
    meteo_cols = ["temperature", "humidity", "cloud_cover", "wind_speed",
                  "heating_degree", "cooling_degree"]
    if include_meteo:
        for col in meteo_cols:
            if col in df_feat.columns:
                feature_cols.append(col)
    
    # Filtrar features existentes
    feature_cols = [c for c in feature_cols if c in df_feat.columns]
    
    # Split
    train = df_feat[df_feat.index <= train_end].copy()
    test = df_feat[(df_feat.index >= forecast_start) & (df_feat.index <= forecast_end)].copy()
    
    if len(test) == 0:
        return None
    
    # Preencher NaN nos lags com valores default
    for col in ["lag_720", "lag_8760"]:
        if col in train.columns:
            train[col] = train[col].fillna(train[col].median() if train[col].notna().any() else 0)
        if col in test.columns:
            # Para teste, usar mediana do treino
            median_val = train[col].median() if train[col].notna().any() else 0
            test[col] = test[col].fillna(median_val)
    
    # Preencher profiles com mediana
    for col in ["hourly_profile", "weekly_profile", "monthly_profile"]:
        if col in train.columns:
            train[col] = train[col].fillna(train[target_col].mean())
        if col in test.columns:
            test[col] = test[col].fillna(train[target_col].mean() if len(train) > 0 else 0)
    
    X_train = train[feature_cols]
    y_train = train.loc[X_train.index, target_col]
    X_test = test[feature_cols]
    y_test = test[target_col]
    
    if len(X_train) < 100:
        print(f"   ⚠️ Treino insuficiente: {len(X_train)} registos")
        return None
    
    # Treinar modelo
    model = HistGradientBoostingRegressor(
        max_depth=8,
        learning_rate=0.05,
        max_iter=300,
        random_state=42
    )
    model.fit(X_train, y_train)
    
    # Prever
    predictions = model.predict(X_test)
    predictions = np.maximum(0, predictions)  # Consumo não pode ser negativo
    
    # Métricas
    mae = mean_absolute_error(y_test, predictions)
    rmse_val = rmse(y_test, predictions)
    total_true = y_test.sum()
    total_pred = predictions.sum()
    erro_pct = abs(total_pred - total_true) / total_true * 100
    
    return {
        "train_end": train_end,
        "forecast_month": forecast_start.strftime("%Y-%m"),
        "n_train": len(train),
        "n_test": len(test),
        "MAE": mae,
        "RMSE": rmse_val,
        "total_true_kwh": total_true,
        "total_pred_kwh": total_pred,
        "erro_pct": erro_pct,
        "season": get_season(forecast_start.month),
        "features_used": len(feature_cols)
    }


def run_real_production_evaluation(df, target_col="target_kwh_hour"):
    """
    Avaliação completa em cenário de produção real.
    """
    
    print("="*80)
    print("🏭 AVALIAÇÃO MULTI-STEP REAL - PREVISÃO DE MÊS INTEIRO")
    print("="*80)
    
    print("""
⚠️  CENÁRIO DE PRODUÇÃO:
    
    Estamos no final do mês M.
    Queremos prever TODO o consumo do mês M+1.
    
    ✅ O que podemos usar:
       - Histórico até fim do mês M
       - Lag de 1 mês (720h) - consumo do mês M
       - Lag de 1 ano (8760h) - mesmo mês do ano passado
       - Perfis médios (hora, semana, mês)
       - Calendário (hora, dia, mês) do mês M+1
       - Previsão meteorológica (ou média histórica)
    
    ❌ O que NÃO podemos usar:
       - lag_1 (hora anterior) - NÃO EXISTE!
       - lag_24 (dia anterior) - NÃO EXISTE!
       - lag_168 (semana anterior) - NÃO EXISTE!
    """)
    
    # Detectar período disponível
    start = df.index.min()
    end = df.index.max()
    
    print(f"\n📅 Dataset: {start.date()} → {end.date()}")
    
    # Vamos testar vários meses
    # Começar depois de ter pelo menos 6 meses de histórico (mínimo razoável)
    # Idealmente 1 ano para ter lag_8760
    
    min_history = pd.DateOffset(months=6)  # Mínimo 6 meses de histórico
    first_forecast = start + min_history
    first_forecast = first_forecast.replace(day=1, hour=0, minute=0, second=0)
    
    results = []
    
    # Gerar lista de meses para testar
    current = first_forecast
    while current + pd.DateOffset(months=1) <= end:
        
        # Treino até fim do mês anterior
        train_end = current - pd.Timedelta(hours=1)
        
        result = forecast_month_real(
            df, target_col, train_end, current, include_meteo=True
        )
        
        if result:
            results.append(result)
            print(f"\n📆 {result['forecast_month']} ({result['season']})")
            print(f"   Treino até: {train_end.date()}")
            print(f"   MAE: {result['MAE']:.4f} kWh")
            print(f"   Total Real:     {result['total_true_kwh']:.0f} kWh")
            print(f"   Total Previsto: {result['total_pred_kwh']:.0f} kWh")
            print(f"   Erro Mensal:    {result['erro_pct']:.2f}%")
        
        current = current + pd.DateOffset(months=1)
    
    if not results:
        print("\n❌ Sem dados suficientes para avaliação")
        return None
    
    # =========================================================================
    # RESUMO
    # =========================================================================
    
    results_df = pd.DataFrame(results)
    
    print("\n" + "="*80)
    print("📊 RESUMO - PREVISÃO MULTI-STEP REAL")
    print("="*80)
    
    print(f"\nMeses avaliados: {len(results_df)}")
    print(f"MAE médio: {results_df['MAE'].mean():.4f} kWh")
    print(f"Erro mensal médio: {results_df['erro_pct'].mean():.2f}%")
    print(f"Erro mensal mínimo: {results_df['erro_pct'].min():.2f}%")
    print(f"Erro mensal máximo: {results_df['erro_pct'].max():.2f}%")
    
    # Por estação
    print("\n📈 Por Estação:")
    for season in ["Inverno", "Primavera", "Verão", "Outono"]:
        season_data = results_df[results_df["season"] == season]
        if len(season_data) > 0:
            print(f"   {season}: {season_data['erro_pct'].mean():.2f}% (n={len(season_data)})")
    
    # Comparação com one-step
    print("\n" + "="*80)
    print("📊 COMPARAÇÃO: ONE-STEP vs MULTI-STEP REAL")
    print("="*80)
    
    print("""
    ┌─────────────────────┬───────────────┬───────────────────┐
    │ Métrica             │ One-Step      │ Multi-Step Real   │
    │                     │ (com lag_1)   │ (sem lags curtos) │
    ├─────────────────────┼───────────────┼───────────────────┤""")
    
    mean_erro = results_df['erro_pct'].mean()
    mean_mae = results_df['MAE'].mean()
    
    print(f"    │ MAE médio           │ ~0.35 kWh     │ {mean_mae:.2f} kWh          │")
    print(f"    │ Erro mensal médio   │ ~1%           │ {mean_erro:.1f}%              │")
    print("""    └─────────────────────┴───────────────┴───────────────────┘
    
    ⚠️  A diferença mostra o impacto REAL dos lags curtos (lag_1, lag_24)
        na performance do modelo!
    """)
    
    return results_df


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    
    # Configuração
    TARGET = "target_kwh_hour"
    
    # Tentar carregar dados
    try:
        # Opção 1: merged_df com meteorologia
        df = pd.read_csv("merged_with_weather.csv", parse_dates=["Datetime"], index_col="Datetime")
        print("✅ Carregado merged_with_weather.csv")
    except:
        try:
            # Opção 2: ficheiro alternativo
            df = pd.read_csv("hourly_consumption.csv", parse_dates=["Datetime"], index_col="Datetime")
            print("✅ Carregado hourly_consumption.csv")
        except:
            # Fallback: sample
            print("⚠️ A usar dataset de amostra (10K registos)")
            df = pd.read_csv("/mnt/project/df_sample_10000.csv", parse_dates=["Datetime"])
            df = df.set_index("Datetime").sort_index()
    
    print(f"📊 Dataset: {len(df)} registos")
    print(f"📅 Período: {df.index.min().date()} → {df.index.max().date()}")
    
    # Correr avaliação
    results = run_real_production_evaluation(df, TARGET)
    
    # Guardar resultados
    if results is not None:
        results.to_csv("multistep_real_results.csv", index=False)
        print("\n✅ Resultados guardados em multistep_real_results.csv")
