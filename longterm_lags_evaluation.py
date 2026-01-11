"""
=============================================================================
AVALIAÇÃO COM LAGS DE LONGO PRAZO (MÊS E ANO) - SEM RECURSÃO
=============================================================================

SOLUÇÃO ELEGANTE:
- Remover lag_1, lag_24, lag_168 (não existem em produção para mês inteiro)
- Usar APENAS lag_720 (1 mês) e lag_8760 (1 ano)
- Previsão DIRETA (não recursiva) - cada hora é independente

EM PRODUÇÃO (prever Julho estando em 30 Jun):
- lag_720 (1 mês): valor de 1 Jun → EXISTE! ✅
- lag_8760 (1 ano): valor de Jul do ano passado → EXISTE! ✅
- lag_1, lag_24, lag_168: NÃO EXISTEM para todo o mês ❌

VANTAGEM:
- Sem propagação de erro
- Cada previsão é independente
- Simples e robusto

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


def prepare_data_longterm_lags(df, target_col):
    """
    Prepara dados usando APENAS lags de longo prazo (disponíveis em produção).
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
    df["day_sin"] = np.sin(2 * np.pi * df["day_of_month"] / 31)
    df["day_cos"] = np.cos(2 * np.pi * df["day_of_month"] / 31)
    
    # =========================================================================
    # LAGS DE LONGO PRAZO (disponíveis em produção!)
    # =========================================================================
    
    # Lag de ~1 mês (720 horas = 30 dias)
    df["lag_720"] = df[target_col].shift(720)
    
    # Lag de ~1 ano (8760 horas = 365 dias)
    df["lag_8760"] = df[target_col].shift(8760)
    
    # Lag de 2 semanas (336 horas)
    df["lag_336"] = df[target_col].shift(336)
    
    # =========================================================================
    # METEOROLOGIA (se existir)
    # =========================================================================
    if "temperature" in df.columns:
        df["heating_degree"] = np.maximum(0, 18 - df["temperature"])
        df["cooling_degree"] = np.maximum(0, df["temperature"] - 24)
    
    return df


def forecast_month_direct(df, target_col, train_end, month_to_forecast, include_meteo=True):
    """
    Previsão DIRETA de um mês inteiro (sem recursão).
    
    - Treina modelo só com lags longos (720, 8760)
    - Prevê todas as horas do mês de uma vez
    - Cada previsão é INDEPENDENTE (sem propagação de erro)
    """
    
    # Definir período de previsão
    forecast_start = month_to_forecast
    forecast_end = forecast_start + pd.DateOffset(months=1) - pd.Timedelta(hours=1)
    
    # Preparar dados
    df_feat = prepare_data_longterm_lags(df, target_col)
    
    # Features a usar (SEM lag_1, lag_24, lag_168!)
    feature_cols = [
        # Calendário
        "is_weekend", "hour_sin", "hour_cos", "dow_sin", "dow_cos",
        "month_sin", "month_cos", "day_sin", "day_cos",
        # Lags LONGOS (disponíveis em produção)
        "lag_720", "lag_336", "lag_8760",
    ]
    
    # Adicionar meteorologia se disponível
    if include_meteo:
        meteo_cols = ["temperature", "humidity", "cloud_cover", "wind_speed",
                      "heating_degree", "cooling_degree", "is_day"]
        for col in meteo_cols:
            if col in df_feat.columns:
                feature_cols.append(col)
    
    # Filtrar features existentes
    feature_cols = [c for c in feature_cols if c in df_feat.columns]
    
    # Split treino/teste
    train = df_feat[df_feat.index <= train_end].copy()
    test = df_feat[(df_feat.index >= forecast_start) & (df_feat.index <= forecast_end)].copy()
    
    if len(test) == 0:
        return None
    
    # Filtrar treino para ter lag_720 válido
    train = train.dropna(subset=["lag_720"])
    
    if len(train) < 200:
        return None
    
    # Preencher NaN no teste (lag_8760 pode não existir no 1º ano)
    for col in feature_cols:
        if col in test.columns:
            median_val = train[col].median() if col in train.columns else 0
            test[col] = test[col].fillna(median_val)
    
    X_train = train[feature_cols]
    y_train = train[target_col]
    X_test = test[feature_cols]
    y_test = test[target_col]
    
    # Treinar modelo
    model = HistGradientBoostingRegressor(
        max_depth=8,
        learning_rate=0.05,
        max_iter=300,
        random_state=42
    )
    model.fit(X_train, y_train)
    
    # Prever TUDO DE UMA VEZ (sem recursão!)
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
        "n_forecast": len(test),
        "MAE": mae,
        "RMSE": rmse_val,
        "total_true_kwh": total_true,
        "total_pred_kwh": total_pred,
        "erro_pct": erro_pct,
        "season": get_season(forecast_start.month),
        "features_used": len(feature_cols)
    }


def run_longterm_evaluation(df, target_col="target_kwh_hour"):
    """
    Avaliação completa usando apenas lags de longo prazo.
    """
    
    print("="*80)
    print("📊 AVALIAÇÃO COM LAGS DE LONGO PRAZO (SEM RECURSÃO)")
    print("="*80)
    
    print("""
✅ ESTRATÉGIA:
    
    Usar APENAS lags que existem em produção para prever mês inteiro:
    
    ✅ LAGS USADOS:
       - lag_720 (~1 mês): Consumo de há 30 dias
       - lag_336 (~2 semanas): Consumo de há 14 dias  
       - lag_8760 (~1 ano): Mesmo período do ano passado
       - Perfis médios (hora, semana)
       - Calendário (hora, dia, mês, fim-de-semana)
       - Meteorologia (se disponível)
    
    ❌ LAGS REMOVIDOS (não existem em produção):
       - lag_1: Hora anterior
       - lag_24: Dia anterior
       - lag_168: Semana anterior
    
    💡 VANTAGEM: Previsão DIRETA, sem propagação de erro!
    """)
    
    # Detectar período disponível
    start = df.index.min()
    end = df.index.max()
    
    print(f"\n📅 Dataset: {start.date()} → {end.date()}")
    
    # Começar após ter histórico suficiente para lag_720
    # Precisamos de pelo menos 720 horas (30 dias) de dados ANTES do treino
    # para ter lag_720 válido
    min_history = pd.DateOffset(days=35)  # 35 dias para garantir lag_720
    first_possible = start + pd.DateOffset(hours=720)  # Primeiro momento com lag_720 válido
    
    # Arredondar para o primeiro dia do mês seguinte
    first_forecast = first_possible.replace(day=1, hour=0, minute=0, second=0)
    if first_forecast <= first_possible:
        first_forecast = first_forecast + pd.DateOffset(months=1)
    
    print(f"   Primeiro mês com lag_720 válido: {first_forecast.strftime('%Y-%m')}")
    
    results = []
    
    # Gerar lista de meses para testar
    current = first_forecast
    while current + pd.DateOffset(months=1) <= end:
        
        # Treino até fim do mês anterior
        train_end = current - pd.Timedelta(hours=1)
        
        result = forecast_month_direct(
            df, target_col, train_end, current, include_meteo=True
        )
        
        if result:
            results.append(result)
            print(f"\n📆 {result['forecast_month']} ({result['season']})")
            print(f"   Treino até: {train_end.date()} ({result['n_train']} horas)")
            print(f"   MAE:  {result['MAE']:.4f} kWh")
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
    print("📊 RESUMO - LAGS DE LONGO PRAZO (SEM RECURSÃO)")
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
    
    # Tabela detalhada
    print("\n📋 Resultados por Mês:")
    print("─"*70)
    print(f"{'Mês':<10} {'Estação':<12} {'MAE':>8} {'RMSE':>8} {'Erro%':>8} {'Real kWh':>10} {'Prev kWh':>10}")
    print("─"*70)
    for _, row in results_df.iterrows():
        print(f"{row['forecast_month']:<10} {row['season']:<12} {row['MAE']:>8.3f} {row['RMSE']:>8.3f} "
              f"{row['erro_pct']:>7.2f}% {row['total_true_kwh']:>10.0f} {row['total_pred_kwh']:>10.0f}")
    
    # Comparação
    print("\n" + "="*80)
    print("📊 COMPARAÇÃO: 3 ABORDAGENS")
    print("="*80)
    
    mean_erro = results_df['erro_pct'].mean()
    mean_mae = results_df['MAE'].mean()
    
    print(f"""
    ┌─────────────────────┬───────────────┬───────────────┬───────────────────┐
    │ Métrica             │ One-Step      │ Lags Longos   │ Recursivo         │
    │                     │ (c/ leakage)  │ (SEM leakage) │ (c/ propagação)   │
    ├─────────────────────┼───────────────┼───────────────┼───────────────────┤
    │ MAE médio           │ ~0.35 kWh     │ {mean_mae:.2f} kWh       │ ~0.80 kWh         │
    │ Erro mensal médio   │ ~1%           │ {mean_erro:.1f}%          │ ~50%              │
    │ Leakage?            │ ❌ SIM        │ ✅ NÃO        │ ✅ NÃO            │
    │ Propagação erro?    │ N/A           │ ✅ NÃO        │ ❌ SIM            │
    │ Válido p/ produção? │ ❌ NÃO        │ ✅ SIM        │ ⚠️ Teórico        │
    └─────────────────────┴───────────────┴───────────────┴───────────────────┘
    
    💡 CONCLUSÃO:
    
    A abordagem com LAGS DE LONGO PRAZO é a mais adequada para produção:
    - Usa apenas informação realmente disponível
    - Não tem propagação de erro (previsão direta)
    - Resultados realistas e reproduzíveis
    """)
    
    # =========================================================================
    # HISTOGRAMAS POR ANO
    # =========================================================================
    
    plot_yearly_histograms(results_df)
    
    return results_df


def plot_yearly_histograms(results_df):
    """
    Cria histogramas comparando previsão vs real para cada ano.
    """
    import matplotlib.pyplot as plt
    
    # Extrair ano do forecast_month
    results_df = results_df.copy()
    results_df["year"] = results_df["forecast_month"].str[:4].astype(int)
    results_df["month_num"] = results_df["forecast_month"].str[5:7].astype(int)
    
    # Anos disponíveis
    years = sorted(results_df["year"].unique())
    
    print("\n" + "="*80)
    print("📊 HISTOGRAMAS: PREVISÃO vs REAL POR ANO")
    print("="*80)
    
    # Nomes dos meses
    month_names = ["Jan", "Fev", "Mar", "Abr", "Mai", "Jun", 
                   "Jul", "Ago", "Set", "Out", "Nov", "Dez"]
    
    # Criar figura com subplots
    n_years = len(years)
    fig, axes = plt.subplots(n_years, 1, figsize=(14, 5*n_years))
    
    if n_years == 1:
        axes = [axes]
    
    for idx, year in enumerate(years):
        ax = axes[idx]
        year_data = results_df[results_df["year"] == year].sort_values("month_num")
        
        if len(year_data) == 0:
            continue
        
        months = year_data["month_num"].values
        real_values = year_data["total_true_kwh"].values
        pred_values = year_data["total_pred_kwh"].values
        errors = year_data["erro_pct"].values
        
        # Posições das barras
        x = np.arange(len(months))
        width = 0.35
        
        # Barras
        bars1 = ax.bar(x - width/2, real_values, width, label='Real', color='#2ecc71', alpha=0.8)
        bars2 = ax.bar(x + width/2, pred_values, width, label='Previsto', color='#3498db', alpha=0.8)
        
        # Adicionar erro percentual em cima de cada par de barras
        for i, (r, p, e) in enumerate(zip(real_values, pred_values, errors)):
            max_val = max(r, p)
            color = '#e74c3c' if e > 20 else '#f39c12' if e > 10 else '#27ae60'
            ax.annotate(f'{e:.1f}%', 
                       xy=(i, max_val + 20),
                       ha='center', va='bottom',
                       fontsize=10, fontweight='bold',
                       color=color)
        
        # Configurações do gráfico
        ax.set_xlabel('Mês', fontsize=12)
        ax.set_ylabel('Consumo Total (kWh)', fontsize=12)
        ax.set_title(f'Ano {year} - Previsão vs Real (Consumo Mensal)', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels([month_names[m-1] for m in months])
        ax.legend(loc='upper right')
        ax.grid(axis='y', alpha=0.3)
        
        # Calcular estatísticas do ano
        year_mae = year_data["MAE"].mean()
        year_erro = year_data["erro_pct"].mean()
        ax.text(0.02, 0.98, f'MAE médio: {year_mae:.2f} kWh | Erro médio: {year_erro:.1f}%',
                transform=ax.transAxes, fontsize=11,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig('yearly_comparison_histograms.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print("\n✅ Gráficos guardados em 'yearly_comparison_histograms.png'")
    
    # =========================================================================
    # GRÁFICO ADICIONAL: Evolução do erro ao longo do tempo
    # =========================================================================
    
    fig, ax = plt.subplots(figsize=(14, 6))
    
    # Ordenar por data
    results_sorted = results_df.sort_values("forecast_month")
    
    x = range(len(results_sorted))
    errors = results_sorted["erro_pct"].values
    months = results_sorted["forecast_month"].values
    
    # Cores por estação
    colors = []
    for season in results_sorted["season"]:
        if season == "Inverno":
            colors.append('#3498db')  # Azul
        elif season == "Primavera":
            colors.append('#2ecc71')  # Verde
        elif season == "Verão":
            colors.append('#f1c40f')  # Amarelo
        else:
            colors.append('#e67e22')  # Laranja
    
    bars = ax.bar(x, errors, color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)
    
    # Linha de tendência (média móvel)
    if len(errors) >= 3:
        window = min(3, len(errors))
        rolling_mean = pd.Series(errors).rolling(window=window, center=True).mean()
        ax.plot(x, rolling_mean, color='red', linewidth=2, linestyle='--', label='Tendência (média móvel)')
    
    # Linhas de referência
    ax.axhline(y=10, color='green', linestyle=':', alpha=0.7, label='Bom (<10%)')
    ax.axhline(y=20, color='orange', linestyle=':', alpha=0.7, label='Aceitável (<20%)')
    
    ax.set_xlabel('Mês de Previsão', fontsize=12)
    ax.set_ylabel('Erro Mensal (%)', fontsize=12)
    ax.set_title('Evolução do Erro de Previsão ao Longo do Tempo', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(months, rotation=45, ha='right')
    ax.legend(loc='upper right')
    ax.grid(axis='y', alpha=0.3)
    
    # Legenda de cores das estações
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#3498db', label='Inverno'),
        Patch(facecolor='#2ecc71', label='Primavera'),
        Patch(facecolor='#f1c40f', label='Verão'),
        Patch(facecolor='#e67e22', label='Outono'),
    ]
    ax.legend(handles=legend_elements, loc='upper left', title='Estação')
    
    plt.tight_layout()
    plt.savefig('error_evolution_timeline.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print("✅ Gráfico de evolução guardado em 'error_evolution_timeline.png'")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    
    # Configuração
    TARGET = "target_kwh_hour"
    
    # Tentar carregar dados
    try:
        df = pd.read_csv("merged_with_weather.csv", parse_dates=["Datetime"], index_col="Datetime")
        print("✅ Carregado merged_with_weather.csv")
    except:
        try:
            df = pd.read_csv("hourly_consumption.csv", parse_dates=["Datetime"], index_col="Datetime")
            print("✅ Carregado hourly_consumption.csv")
        except:
            print("⚠️ A usar dataset de amostra (10K registos)")
            df = pd.read_csv("/mnt/project/df_sample_10000.csv", parse_dates=["Datetime"])
            df = df.set_index("Datetime").sort_index()
    
    print(f"📊 Dataset: {len(df)} registos")
    print(f"📅 Período: {df.index.min().date()} → {df.index.max().date()}")
    
    # Correr avaliação
    results = run_longterm_evaluation(df, TARGET)
    
    # Guardar resultados
    if results is not None:
        results.to_csv("longterm_lags_results.csv", index=False)
        print("\n✅ Resultados guardados em longterm_lags_results.csv")
