"""
=============================================================================
AVALIAÇÃO PROGRESSIVA - SIMULAÇÃO DE PRODUÇÃO
=============================================================================

Objetivo: Ver como o modelo melhora à medida que acumula mais dados de treino
e como se comporta em diferentes estações do ano.

Cenários de teste:
1. Treino: 9 meses  → Teste: mês seguinte
2. Treino: 1.5 anos → Teste: mês seguinte  
3. Treino: 2 anos 8 meses → Teste: mês seguinte (Estação 1)
4. Treino: 3 anos 1 mês   → Teste: mês seguinte (Estação 2)
5. Treino: 3 anos 4 meses → Teste: mês seguinte (Estação 3)
6. Treino: 3 anos 8 meses → Teste: mês seguinte (Estação 4)

=============================================================================
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

# Tentar importar LightGBM (melhor performance)
try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False


def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


def get_season(month):
    """Retorna a estação do ano."""
    if month in [12, 1, 2]:
        return "Inverno"
    elif month in [3, 4, 5]:
        return "Primavera"
    elif month in [6, 7, 8]:
        return "Verão"
    else:
        return "Outono"


def create_features(df, target_col="target_kwh_hour"):
    """
    Cria features para o modelo.
    Lags ficam com NaN onde não existem (modelos suportam).
    """
    df = df.copy()
    
    # Calendário
    df["hour"] = df.index.hour
    df["day_of_week"] = df.index.dayofweek
    df["is_weekend"] = (df.index.dayofweek >= 5).astype(int)
    
    # Encoding cíclico
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["dow_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)
    df["month_sin"] = np.sin(2 * np.pi * df.index.month / 12)
    df["month_cos"] = np.cos(2 * np.pi * df.index.month / 12)
    
    # Lags (NaN onde não existem)
    for lag in [1, 24, 168]:
        df[f"lag_{lag}"] = df[target_col].shift(lag)
    
    # Meteorologia derivada (se existir)
    if "temperature" in df.columns:
        df["heating_degree"] = np.maximum(0, 18 - df["temperature"])
        df["cooling_degree"] = np.maximum(0, df["temperature"] - 24)
    
    return df


def get_feature_columns(df):
    """Retorna lista de colunas de features."""
    base_features = [
        "is_weekend", "hour_sin", "hour_cos", "dow_sin", "dow_cos",
        "month_sin", "month_cos", "lag_1", "lag_24", "lag_168"
    ]
    
    # Adicionar meteo se existir
    meteo_cols = ["temperature", "humidity", "precipitation", "cloud_cover",
                  "wind_speed", "is_day", "heating_degree", "cooling_degree"]
    
    for col in meteo_cols:
        if col in df.columns:
            base_features.append(col)
    
    return [c for c in base_features if c in df.columns]


def train_and_evaluate(df, train_end, test_start, test_end, target_col, feature_cols):
    """
    Treina modelo até train_end e avalia no período test_start:test_end.
    
    Returns:
        dict com métricas
    """
    # Split
    train = df[df.index <= train_end]
    test = df[(df.index >= test_start) & (df.index <= test_end)]
    
    if len(test) == 0:
        return None
    
    X_train = train[feature_cols]
    y_train = train[target_col]
    X_test = test[feature_cols]
    y_test = test[target_col]
    
    # Modelo
    if HAS_LIGHTGBM:
        model = lgb.LGBMRegressor(
            n_estimators=300,
            max_depth=8,
            learning_rate=0.05,
            random_state=42,
            verbosity=-1
        )
    else:
        model = HistGradientBoostingRegressor(
            max_depth=8,
            learning_rate=0.05,
            max_iter=300,
            random_state=42
        )
    
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    
    # Métricas
    mae = mean_absolute_error(y_test, pred)
    rmse_val = rmse(y_test, pred)
    
    total_true = y_test.sum()
    total_pred = pred.sum()
    erro_pct = abs(total_pred - total_true) / total_true * 100
    
    return {
        "n_train": len(train),
        "n_test": len(test),
        "MAE": mae,
        "RMSE": rmse_val,
        "total_true_kwh": total_true,
        "total_pred_kwh": total_pred,
        "erro_pct": erro_pct
    }


def progressive_evaluation(df, target_col="target_kwh_hour"):
    """
    Avaliação progressiva: treina com cada vez mais dados e testa o mês seguinte.
    """
    # Criar features
    df = create_features(df, target_col)
    feature_cols = get_feature_columns(df)
    
    start = df.index.min()
    end = df.index.max()
    
    print("="*80)
    print("📊 AVALIAÇÃO PROGRESSIVA - SIMULAÇÃO DE PRODUÇÃO")
    print("="*80)
    print(f"\nDataset: {start.date()} → {end.date()}")
    print(f"Features: {len(feature_cols)}")
    print(f"Modelo: {'LightGBM' if HAS_LIGHTGBM else 'HistGradientBoosting'}")
    
    # =========================================================================
    # DEFINIR CENÁRIOS DE TESTE
    # =========================================================================
    
    scenarios = [
        {
            "name": "9 meses de treino",
            "train_months": 9,
            "description": "Modelo 'jovem' - pouco histórico"
        },
        {
            "name": "1.5 anos de treino", 
            "train_months": 18,
            "description": "Modelo com um ciclo sazonal parcial"
        },
        {
            "name": "2 anos 8 meses (Outono)",
            "train_months": 32,
            "description": "Estação 1 - Testar Outono"
        },
        {
            "name": "3 anos 1 mês (Inverno)",
            "train_months": 37,
            "description": "Estação 2 - Testar Inverno"
        },
        {
            "name": "3 anos 4 meses (Primavera)",
            "train_months": 40,
            "description": "Estação 3 - Testar Primavera"
        },
        {
            "name": "3 anos 8 meses (Verão)",
            "train_months": 44,
            "description": "Estação 4 - Testar Verão"
        },
    ]
    
    results = []
    
    print("\n" + "─"*80)
    
    for scenario in scenarios:
        train_months = scenario["train_months"]
        
        # Calcular datas
        train_end = start + pd.DateOffset(months=train_months) - pd.Timedelta(hours=1)
        test_start = train_end + pd.Timedelta(hours=1)
        test_end = test_start + pd.DateOffset(months=1) - pd.Timedelta(hours=1)
        
        # Verificar se temos dados suficientes
        if test_end > end:
            print(f"\n⚠️ {scenario['name']}: Dados insuficientes (precisaria até {test_end.date()})")
            continue
        
        # Avaliar
        metrics = train_and_evaluate(
            df, train_end, test_start, test_end, target_col, feature_cols
        )
        
        if metrics is None:
            continue
        
        # Adicionar info do cenário
        test_month = test_start.month
        season = get_season(test_month)
        
        metrics["scenario"] = scenario["name"]
        metrics["train_period"] = f"{start.date()} → {train_end.date()}"
        metrics["test_period"] = f"{test_start.date()} → {test_end.date()}"
        metrics["test_month_name"] = test_start.strftime("%B %Y")
        metrics["season"] = season
        
        results.append(metrics)
        
        # Output
        print(f"\n📅 {scenario['name']}")
        print(f"   {scenario['description']}")
        print(f"   Treino: {start.date()} → {train_end.date()} ({metrics['n_train']} horas)")
        print(f"   Teste:  {test_start.date()} → {test_end.date()} ({metrics['n_test']} horas)")
        print(f"   Estação: {season} | Mês: {test_start.strftime('%B %Y')}")
        print(f"   ─────────────────────────────────────")
        print(f"   MAE:  {metrics['MAE']:.4f} kWh")
        print(f"   RMSE: {metrics['RMSE']:.4f} kWh")
        print(f"   Total Real:     {metrics['total_true_kwh']:.0f} kWh")
        print(f"   Total Previsto: {metrics['total_pred_kwh']:.0f} kWh")
        print(f"   Erro Mensal:    {metrics['erro_pct']:.2f}%")
    
    # =========================================================================
    # RESUMO FINAL
    # =========================================================================
    
    if not results:
        print("\n❌ Nenhum cenário pôde ser avaliado (dados insuficientes)")
        return None
    
    results_df = pd.DataFrame(results)
    
    print("\n" + "="*80)
    print("📊 RESUMO FINAL")
    print("="*80)
    
    print("\n┌─────────────────────────────────────┬──────────┬────────────┬───────────┐")
    print("│ Cenário                             │ N_treino │   MAE      │  Erro %   │")
    print("├─────────────────────────────────────┼──────────┼────────────┼───────────┤")
    
    for _, row in results_df.iterrows():
        print(f"│ {row['scenario']:<35} │ {row['n_train']:>8} │ {row['MAE']:>10.4f} │ {row['erro_pct']:>8.2f}% │")
    
    print("└─────────────────────────────────────┴──────────┴────────────┴───────────┘")
    
    # Análise
    print("\n" + "─"*80)
    print("💡 ANÁLISE")
    print("─"*80)
    
    # Evolução do erro com mais dados
    first_error = results_df.iloc[0]["erro_pct"]
    last_error = results_df.iloc[-1]["erro_pct"] if len(results_df) > 1 else first_error
    
    print(f"\n1. EVOLUÇÃO COM MAIS DADOS:")
    print(f"   Erro com {results_df.iloc[0]['n_train']} horas de treino: {first_error:.2f}%")
    if len(results_df) > 1:
        print(f"   Erro com {results_df.iloc[-1]['n_train']} horas de treino: {last_error:.2f}%")
        improvement = first_error - last_error
        print(f"   Melhoria: {improvement:+.2f} pontos percentuais")
    
    # Análise por estação
    print(f"\n2. DESEMPENHO POR ESTAÇÃO:")
    for season in ["Inverno", "Primavera", "Verão", "Outono"]:
        season_data = results_df[results_df["season"] == season]
        if len(season_data) > 0:
            mean_error = season_data["erro_pct"].mean()
            print(f"   {season}: {mean_error:.2f}% erro médio")
    
    # Melhor e pior
    best = results_df.loc[results_df["erro_pct"].idxmin()]
    worst = results_df.loc[results_df["erro_pct"].idxmax()]
    
    print(f"\n3. MELHOR RESULTADO:")
    print(f"   {best['scenario']} → {best['erro_pct']:.2f}% ({best['test_month_name']})")
    
    print(f"\n4. PIOR RESULTADO:")
    print(f"   {worst['scenario']} → {worst['erro_pct']:.2f}% ({worst['test_month_name']})")
    
    return results_df


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    
    # =========================================================================
    # CONFIGURAÇÃO - AJUSTAR PATHS
    # =========================================================================
    
    # Opção 1: Se tens o merged_df já pronto
    # MERGED_PATH = "merged_hourly_weather.csv"
    # df = pd.read_csv(MERGED_PATH, parse_dates=["Datetime"], index_col="Datetime")
    
    # Opção 2: Carregar separadamente e fazer merge
    CONSUMPTION_PATH = "hourly.csv"
    WEATHER_PATH = "weather_sceaux_2006_2010.csv"
    TARGET = "target_kwh_hour"
    
    try:
        print("📂 Carregando dados...")
        consumption = pd.read_csv(CONSUMPTION_PATH, parse_dates=["Datetime"])
        weather = pd.read_csv(WEATHER_PATH, parse_dates=["Datetime"])
        
        df = consumption.merge(weather, on="Datetime", how="left")
        df = df.set_index("Datetime").sort_index()
        
        print(f"✅ Dados carregados: {len(df)} registos")
        print(f"   Período: {df.index.min().date()} → {df.index.max().date()}")
        
    except FileNotFoundError as e:
        print(f"⚠️ Ficheiros não encontrados: {e}")
        print("   A usar dataset de amostra (10K registos)...")
        
        df = pd.read_csv("/mnt/project/df_sample_10000.csv", parse_dates=["Datetime"])
        df = df.set_index("Datetime").sort_index()
        TARGET = "target_kwh_hour"
    
    # =========================================================================
    # CORRER AVALIAÇÃO
    # =========================================================================
    
    results = progressive_evaluation(df, TARGET)
    
    # Guardar resultados
    if results is not None:
        results.to_csv("progressive_evaluation_results.csv", index=False)
        print("\n✅ Resultados guardados em progressive_evaluation_results.csv")
