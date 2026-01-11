"""
=============================================================================
ENSEMBLE: PERSISTÊNCIA SEMANAL + AJUSTE SAZONAL/METEOROLÓGICO
=============================================================================

AUTOR: Álvaro Terroso
CURSO: Machine Learning Algorithms - MAAI - IPCA
DATASET: Individual Household Electric Power Consumption (UCI)
         ~3 anos e 9 meses de dados (Dez 2006 - Nov 2010)

=============================================================================
ESTRATÉGIA
=============================================================================

O ensemble combina duas componentes:

1. PERSISTÊNCIA SEMANAL (lag_168)
   - Base sólida que captura padrões comportamentais
   - Valor da mesma hora, mesmo dia da semana, semana passada
   
2. MODELO DE AJUSTE
   - Aprende a corrigir a persistência com base em:
     * Diferença de temperatura vs semana passada
     * Sazonalidade (mês do ano)
     * Padrões de aquecimento/arrefecimento

PREVISÃO FINAL = Persistência + Ajuste (limitado a ±30% da base)

=============================================================================
INSTRUÇÕES DE USO
=============================================================================

1. Coloca este script na mesma pasta que:
   - O teu dataset horário (CSV com coluna Datetime e target_kwh_hour)
   - O ficheiro de meteorologia (weather_sceaux_2006_2010.csv)

2. Ajusta os paths no final do script se necessário

3. Corre: python ensemble_persistence_seasonal.py

=============================================================================
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
import warnings
warnings.filterwarnings('ignore')


# =============================================================================
# MÉTRICAS
# =============================================================================

def rmse(y_true, y_pred):
    """Root Mean Squared Error"""
    return np.sqrt(mean_squared_error(y_true, y_pred))

def mape(y_true, y_pred):
    """Mean Absolute Percentage Error"""
    mask = y_true != 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100


# =============================================================================
# CARREGAMENTO E PREPARAÇÃO DE DADOS
# =============================================================================

def load_and_merge_data(consumption_path, weather_path):
    """
    Carrega dados de consumo e meteorologia, faz merge por Datetime.
    
    Args:
        consumption_path: Path para CSV com colunas Datetime, target_kwh_hour, etc.
        weather_path: Path para CSV com colunas Datetime, temperature, humidity, etc.
    
    Returns:
        DataFrame merged e ordenado por Datetime
    """
    print("📂 Carregando dados de consumo...")
    df = pd.read_csv(consumption_path, parse_dates=["Datetime"])
    print(f"   → {len(df)} registos carregados")
    
    print("🌤️ Carregando dados meteorológicos...")
    weather = pd.read_csv(weather_path, parse_dates=["Datetime"])
    print(f"   → {len(weather)} registos carregados")
    
    print("🔗 Fazendo merge...")
    df = df.merge(weather, on="Datetime", how="left")
    df = df.sort_values("Datetime").set_index("Datetime")
    
    # Verificar nulls
    nulls = df.isnull().sum()
    if nulls.any():
        print(f"⚠️ Nulls encontrados:\n{nulls[nulls > 0]}")
    
    print(f"✅ Dataset final: {len(df)} registos")
    print(f"📅 Período: {df.index.min()} → {df.index.max()}")
    
    return df


def create_features(df, target_col="target_kwh_hour"):
    """
    Cria todas as features necessárias para o ensemble.
    
    Features criadas:
    - Calendário: hora, dia da semana, mês, etc.
    - Encoding cíclico: sin/cos para periodicidade
    - Lags: 1h, 24h, 168h (1 semana)
    - Meteorologia: temperatura, graus de aquecimento/arrefecimento
    - Diferenças temporais: delta de temperatura vs semana passada
    """
    df = df.copy()
    
    print("\n🔧 Criando features...")
    
    # -------------------------------------------------------------------------
    # Calendário
    # -------------------------------------------------------------------------
    df["hour"] = df.index.hour
    df["day_of_week"] = df.index.dayofweek
    df["month"] = df.index.month
    df["day_of_year"] = df.index.dayofyear
    df["week_of_year"] = df.index.isocalendar().week.astype(int)
    df["is_weekend"] = (df.index.dayofweek >= 5).astype(int)
    
    # -------------------------------------------------------------------------
    # Encoding Cíclico
    # -------------------------------------------------------------------------
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["dow_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    df["doy_sin"] = np.sin(2 * np.pi * df["day_of_year"] / 365)
    df["doy_cos"] = np.cos(2 * np.pi * df["day_of_year"] / 365)
    
    # -------------------------------------------------------------------------
    # Lags do Target
    # -------------------------------------------------------------------------
    df["lag_1"] = df[target_col].shift(1)
    df["lag_24"] = df[target_col].shift(24)
    df["lag_168"] = df[target_col].shift(168)  # 1 semana
    df["lag_336"] = df[target_col].shift(336)  # 2 semanas
    df["lag_8760"] = df[target_col].shift(8760)  # 1 ano (se disponível)
    
    # -------------------------------------------------------------------------
    # Meteorologia Derivada
    # -------------------------------------------------------------------------
    if "temperature" in df.columns:
        # Graus-dia de aquecimento (base 18°C)
        df["heating_degree"] = np.maximum(0, 18 - df["temperature"])
        # Graus-dia de arrefecimento (base 24°C)
        df["cooling_degree"] = np.maximum(0, df["temperature"] - 24)
        
        # Lags de temperatura
        df["temp_lag_168"] = df["temperature"].shift(168)
        df["heating_lag_168"] = df["heating_degree"].shift(168)
        
        # Diferença de temperatura vs semana passada
        df["temp_diff_168"] = df["temperature"] - df["temp_lag_168"]
        df["heating_diff_168"] = df["heating_degree"] - df["heating_lag_168"]
    
    # -------------------------------------------------------------------------
    # Rolling Statistics (para contexto)
    # -------------------------------------------------------------------------
    df["rolling_mean_24h"] = df[target_col].shift(1).rolling(24, min_periods=1).mean()
    df["rolling_mean_168h"] = df[target_col].shift(1).rolling(168, min_periods=1).mean()
    
    # -------------------------------------------------------------------------
    # Remover linhas com NaN (início da série)
    # -------------------------------------------------------------------------
    initial_len = len(df)
    df = df.dropna()
    removed = initial_len - len(df)
    print(f"   → Removidas {removed} linhas iniciais (lags)")
    print(f"   → Dataset final: {len(df)} registos")
    
    return df


# =============================================================================
# MODELO ENSEMBLE
# =============================================================================

def train_adjustment_model(train_data, target_col="target_kwh_hour"):
    """
    Treina o modelo de ajuste que aprende a corrigir a persistência.
    
    O modelo prevê: DELTA = valor_real - lag_168
    
    Baseado em:
    - Diferença de temperatura vs semana passada
    - Calendário (hora, dia, mês)
    - Padrões de aquecimento
    """
    train_data = train_data.copy()
    
    # Target: diferença entre real e persistência
    train_data["delta"] = train_data[target_col] - train_data["lag_168"]
    
    # Features para o modelo de ajuste
    feature_cols = [
        # Calendário
        "hour_sin", "hour_cos",
        "dow_sin", "dow_cos", 
        "month_sin", "month_cos",
        "doy_sin", "doy_cos",
        "is_weekend",
        # Meteorologia atual
        "temperature", "heating_degree", "cooling_degree",
        # Diferenças vs semana passada
        "temp_diff_168", "heating_diff_168",
        # Contexto
        "rolling_mean_24h", "rolling_mean_168h"
    ]
    
    # Filtrar colunas que existem
    feature_cols = [c for c in feature_cols if c in train_data.columns]
    
    X = train_data[feature_cols]
    y = train_data["delta"]
    
    # Modelo com regularização para evitar overfitting
    model = HistGradientBoostingRegressor(
        max_depth=6,
        learning_rate=0.03,
        max_iter=200,
        l2_regularization=1.0,
        random_state=42,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=20
    )
    
    model.fit(X, y)
    
    return model, feature_cols


def forecast_month_ensemble(model, feature_cols, df, train_end, 
                            month_start, month_end, target_col="target_kwh_hour",
                            max_adjustment_pct=0.3):
    """
    Prevê um mês usando o ensemble: Persistência + Ajuste.
    
    Args:
        model: Modelo de ajuste treinado
        feature_cols: Lista de features para o modelo
        df: DataFrame com todos os dados
        train_end: Último timestamp disponível para treino
        month_start, month_end: Período a prever
        target_col: Nome da coluna target
        max_adjustment_pct: Ajuste máximo como fração da base (default 30%)
    
    Returns:
        Series com previsões, Series com componente de persistência
    """
    forecast_range = pd.date_range(month_start, month_end, freq="h")
    
    # Histórico até train_end (real)
    history = df.loc[:train_end].copy()
    
    predictions = []
    persistence_values = []
    
    for ts in forecast_range:
        # -----------------------------------------------------------------
        # 1. PERSISTÊNCIA: valor da mesma hora há 1 semana
        # -----------------------------------------------------------------
        lag_ts = ts - pd.Timedelta(hours=168)
        
        if lag_ts in history.index:
            persistence = history.loc[lag_ts, target_col]
            temp_lag = history.loc[lag_ts, "temperature"] if "temperature" in history.columns else 15
            heating_lag = history.loc[lag_ts, "heating_degree"] if "heating_degree" in history.columns else 5
        else:
            persistence = history[target_col].mean()
            temp_lag = history["temperature"].mean() if "temperature" in history.columns else 15
            heating_lag = history["heating_degree"].mean() if "heating_degree" in history.columns else 5
        
        persistence_values.append(persistence)
        
        # -----------------------------------------------------------------
        # 2. FEATURES para ajuste
        # -----------------------------------------------------------------
        # Meteorologia do timestamp a prever
        if ts in df.index:
            temp_now = df.loc[ts, "temperature"]
            heating_now = df.loc[ts, "heating_degree"]
            cooling_now = df.loc[ts, "cooling_degree"]
        else:
            temp_now = df["temperature"].mean()
            heating_now = df["heating_degree"].mean()
            cooling_now = df["cooling_degree"].mean()
        
        features = {
            "hour_sin": np.sin(2 * np.pi * ts.hour / 24),
            "hour_cos": np.cos(2 * np.pi * ts.hour / 24),
            "dow_sin": np.sin(2 * np.pi * ts.dayofweek / 7),
            "dow_cos": np.cos(2 * np.pi * ts.dayofweek / 7),
            "month_sin": np.sin(2 * np.pi * ts.month / 12),
            "month_cos": np.cos(2 * np.pi * ts.month / 12),
            "doy_sin": np.sin(2 * np.pi * ts.dayofyear / 365),
            "doy_cos": np.cos(2 * np.pi * ts.dayofyear / 365),
            "is_weekend": int(ts.dayofweek >= 5),
            "temperature": temp_now,
            "heating_degree": heating_now,
            "cooling_degree": cooling_now,
            "temp_diff_168": temp_now - temp_lag,
            "heating_diff_168": heating_now - heating_lag,
            "rolling_mean_24h": history[target_col].tail(24).mean(),
            "rolling_mean_168h": history[target_col].tail(168).mean()
        }
        
        # -----------------------------------------------------------------
        # 3. AJUSTE via modelo
        # -----------------------------------------------------------------
        X = pd.DataFrame([features])[[c for c in feature_cols if c in features]]
        adjustment = model.predict(X)[0]
        
        # Limitar ajuste a ±X% da persistência
        max_adj = abs(persistence) * max_adjustment_pct
        adjustment = np.clip(adjustment, -max_adj, max_adj)
        
        # -----------------------------------------------------------------
        # 4. PREVISÃO FINAL
        # -----------------------------------------------------------------
        pred = persistence + adjustment
        pred = max(0, pred)  # Consumo não pode ser negativo
        predictions.append(pred)
        
        # Atualizar histórico com previsão para próximas iterações
        if ts not in history.index:
            history.loc[ts, target_col] = pred
            history.loc[ts, "temperature"] = temp_now
            history.loc[ts, "heating_degree"] = heating_now
    
    return pd.Series(predictions, index=forecast_range), pd.Series(persistence_values, index=forecast_range)


# =============================================================================
# WALK-FORWARD VALIDATION
# =============================================================================

def walk_forward_validation(df, first_month, n_months, target_col="target_kwh_hour"):
    """
    Validação walk-forward: treina até mês M-1, prevê mês M.
    
    Simula cenário de produção onde:
    - Temos dados reais até ao fim do mês atual
    - Queremos prever o próximo mês completo
    - Após o mês acontecer, os dados reais ficam disponíveis para retreino
    
    Args:
        df: DataFrame com features
        first_month: Primeiro mês a prever (ex: "2009-07")
        n_months: Quantos meses prever
        target_col: Nome do target
    
    Returns:
        DataFrame com resultados detalhados por mês
    """
    results = []
    
    first_month_start = pd.Timestamp(first_month + "-01")
    current_train_end = first_month_start - pd.Timedelta(hours=1)
    
    print("\n" + "="*70)
    print("📊 WALK-FORWARD VALIDATION")
    print("="*70)
    
    for i in range(n_months):
        month_start = first_month_start + pd.DateOffset(months=i)
        month_end = (month_start + pd.offsets.MonthEnd(1)).normalize() + pd.Timedelta(hours=23)
        month_str = month_start.strftime("%Y-%m")
        
        # Verificar se temos dados
        if month_end > df.index.max():
            print(f"\n⚠️ Dados insuficientes para {month_str}. Parando.")
            break
        
        print(f"\n{'─'*70}")
        print(f"📅 Mês: {month_str}")
        print(f"🏋️ Treino até: {current_train_end}")
        
        # Dados de treino
        train_data = df[df.index <= current_train_end]
        print(f"📊 Registos de treino: {len(train_data)}")
        
        # Treinar modelo de ajuste
        adj_model, feature_cols = train_adjustment_model(train_data, target_col)
        
        # Prever mês com ensemble
        y_pred_ens, y_pred_persist = forecast_month_ensemble(
            adj_model, feature_cols, df, current_train_end,
            month_start, month_end, target_col
        )
        
        # Ground truth
        y_true = df.loc[month_start:month_end, target_col]
        
        # Alinhar índices
        common_idx = y_pred_ens.index.intersection(y_true.index)
        y_pred_ens = y_pred_ens.loc[common_idx]
        y_pred_persist = y_pred_persist.loc[common_idx]
        y_true = y_true.loc[common_idx]
        
        # Métricas - Ensemble
        total_true = y_true.sum()
        total_ens = y_pred_ens.sum()
        pct_err_ens = abs(total_ens - total_true) / total_true * 100
        mae_ens = mean_absolute_error(y_true, y_pred_ens)
        rmse_ens = rmse(y_true, y_pred_ens)
        
        # Métricas - Persistência pura (para comparação)
        total_persist = y_pred_persist.sum()
        pct_err_persist = abs(total_persist - total_true) / total_true * 100
        mae_persist = mean_absolute_error(y_true, y_pred_persist)
        
        results.append({
            "month": month_str,
            "n_hours": len(common_idx),
            "total_true_kwh": round(total_true, 2),
            "total_ensemble_kwh": round(total_ens, 2),
            "total_persist_kwh": round(total_persist, 2),
            "pct_error_ensemble": round(pct_err_ens, 2),
            "pct_error_persist": round(pct_err_persist, 2),
            "mae_ensemble": round(mae_ens, 4),
            "mae_persist": round(mae_persist, 4),
            "rmse_ensemble": round(rmse_ens, 4),
            "improvement_pp": round(pct_err_persist - pct_err_ens, 2)
        })
        
        # Output
        winner = "Ensemble ✓" if pct_err_ens < pct_err_persist else "Persist ✓"
        print(f"📈 Real: {total_true:.0f} kWh")
        print(f"🎯 Ensemble: {total_ens:.0f} kWh (erro: {pct_err_ens:.2f}%)")
        print(f"📌 Persist:  {total_persist:.0f} kWh (erro: {pct_err_persist:.2f}%)")
        print(f"🏆 Melhor: {winner}")
        
        # Avançar train_end (dados reais do mês ficam disponíveis)
        current_train_end = month_end
    
    return pd.DataFrame(results)


# =============================================================================
# ANÁLISE DE RESULTADOS
# =============================================================================

def analyze_results(results_df):
    """Gera análise detalhada dos resultados."""
    
    print("\n" + "="*70)
    print("📊 ANÁLISE DE RESULTADOS")
    print("="*70)
    
    # Tabela resumo
    print("\n📋 RESULTADOS POR MÊS:")
    print("─"*90)
    print(results_df[["month", "total_true_kwh", "total_ensemble_kwh", 
                      "pct_error_ensemble", "pct_error_persist", "improvement_pp"]].to_string(index=False))
    
    # Métricas agregadas
    print("\n" + "─"*70)
    print("📈 MÉTRICAS AGREGADAS:")
    print("─"*70)
    
    mean_err_ens = results_df["pct_error_ensemble"].mean()
    mean_err_persist = results_df["pct_error_persist"].mean()
    std_err_ens = results_df["pct_error_ensemble"].std()
    
    print(f"   Erro médio Ensemble:     {mean_err_ens:.2f}% (±{std_err_ens:.2f}%)")
    print(f"   Erro médio Persistência: {mean_err_persist:.2f}%")
    print(f"   Melhoria média:          {mean_err_persist - mean_err_ens:+.2f} pp")
    
    # MAE
    mean_mae_ens = results_df["mae_ensemble"].mean()
    mean_mae_persist = results_df["mae_persist"].mean()
    print(f"\n   MAE médio Ensemble:      {mean_mae_ens:.4f} kWh")
    print(f"   MAE médio Persistência:  {mean_mae_persist:.4f} kWh")
    
    # Contagem de vitórias
    wins_ens = (results_df["improvement_pp"] > 0).sum()
    wins_persist = (results_df["improvement_pp"] <= 0).sum()
    print(f"\n🏆 Vitórias: Ensemble={wins_ens} | Persistência={wins_persist}")
    
    # Análise sazonal
    print("\n" + "─"*70)
    print("🌡️ ANÁLISE POR ESTAÇÃO:")
    print("─"*70)
    
    results_df["month_num"] = results_df["month"].str[-2:].astype(int)
    
    # Inverno: Dez, Jan, Fev
    winter = results_df[results_df["month_num"].isin([12, 1, 2])]
    # Primavera: Mar, Abr, Mai
    spring = results_df[results_df["month_num"].isin([3, 4, 5])]
    # Verão: Jun, Jul, Ago
    summer = results_df[results_df["month_num"].isin([6, 7, 8])]
    # Outono: Set, Out, Nov
    autumn = results_df[results_df["month_num"].isin([9, 10, 11])]
    
    for name, season_df in [("Inverno", winter), ("Primavera", spring), 
                            ("Verão", summer), ("Outono", autumn)]:
        if len(season_df) > 0:
            print(f"   {name}: Erro médio = {season_df['pct_error_ensemble'].mean():.2f}% "
                  f"(n={len(season_df)} meses)")
    
    return {
        "mean_error_ensemble": mean_err_ens,
        "mean_error_persist": mean_err_persist,
        "improvement": mean_err_persist - mean_err_ens,
        "wins_ensemble": wins_ens,
        "wins_persist": wins_persist
    }


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    
    print("="*70)
    print("🔌 ENSEMBLE: PERSISTÊNCIA + AJUSTE SAZONAL/METEOROLÓGICO")
    print("="*70)
    
    # =========================================================================
    # CONFIGURAÇÃO - AJUSTAR CONFORME NECESSÁRIO
    # =========================================================================
    
    # Paths para os ficheiros (ajustar se necessário)
    CONSUMPTION_PATH = "hourly.csv"  # O teu CSV com dados horários
    WEATHER_PATH = "weather_sceaux_2006_2010.csv"  # Meteorologia real
    
    # Coluna target
    TARGET_COL = "target_kwh_hour"
    
    # Período de validação
    # Com ~4 anos de dados, sugiro:
    # - Treino inicial: 2 anos (Dez 2006 - Nov 2008)
    # - Validação: 1 ano (Dez 2008 - Nov 2009)
    # - Teste: ~1 ano (Dez 2009 - Nov 2010)
    
    FIRST_VALIDATION_MONTH = "2010-1"  # Primeiro mês a prever na validação
    N_VALIDATION_MONTHS = 6  # 12 meses de validação
    
    FIRST_TEST_MONTH = "2009-12"  # Primeiro mês do teste
    N_TEST_MONTHS = 12  # 12 meses de teste (ou até acabarem os dados)
    
    # =========================================================================
    # EXECUÇÃO
    # =========================================================================
    
    try:
        # 1. Carregar dados
        df = load_and_merge_data(CONSUMPTION_PATH, WEATHER_PATH)
        
        # 2. Criar features
        df = create_features(df, TARGET_COL)
        
        # 3. Walk-forward VALIDAÇÃO
        print("\n" + "="*70)
        print("🔬 FASE 1: VALIDAÇÃO")
        print("="*70)
        
        results_val = walk_forward_validation(
            df, 
            first_month=FIRST_VALIDATION_MONTH,
            n_months=N_VALIDATION_MONTHS,
            target_col=TARGET_COL
        )
        
        summary_val = analyze_results(results_val)
        '''
        
        # 4. Walk-forward TESTE (se quiser testar no período reservado)
        print("\n" + "="*70)
        print("🧪 FASE 2: TESTE")
        print("="*70)
        

        results_test = walk_forward_validation(
            df,
            first_month=FIRST_TEST_MONTH,
            n_months=N_TEST_MONTHS,
            target_col=TARGET_COL
        )
        
        summary_test = analyze_results(results_test)
        
        # 5. Guardar resultados
        results_val.to_csv("results_validation.csv", index=False)
        results_test.to_csv("results_test.csv", index=False)
        print("\n✅ Resultados guardados em results_validation.csv e results_test.csv")
        
        # 6. Conclusão
        print("\n" + "="*70)
        print("📋 CONCLUSÃO FINAL")
        print("="*70)
        print(f"""
Dataset: ~{len(df)} registos horários ({df.index.min().date()} a {df.index.max().date()})

VALIDAÇÃO ({N_VALIDATION_MONTHS} meses):
   Erro médio Ensemble:     {summary_val['mean_error_ensemble']:.2f}%
   Erro médio Persistência: {summary_val['mean_error_persist']:.2f}%
   Vitórias Ensemble:       {summary_val['wins_ensemble']}/{N_VALIDATION_MONTHS}

TESTE ({len(results_test)} meses):
   Erro médio Ensemble:     {summary_test['mean_error_ensemble']:.2f}%
   Erro médio Persistência: {summary_test['mean_error_persist']:.2f}%
   Vitórias Ensemble:       {summary_test['wins_ensemble']}/{len(results_test)}
""")

'''
        
    except FileNotFoundError as e:
        print(f"\n❌ Erro: Ficheiro não encontrado - {e}")
        print("\n📝 Instruções:")
        print("1. Certifica-te que o CSV de consumo está no path correto")
        print("2. Certifica-te que o CSV de meteorologia está no path correto")
        print("3. Ajusta as variáveis CONSUMPTION_PATH e WEATHER_PATH no script")
        
    except Exception as e:
        print(f"\n❌ Erro: {e}")
        import traceback
        traceback.print_exc()
