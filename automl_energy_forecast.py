"""
=============================================================================
AUTOML PARA PREVISÃO DE CONSUMO ENERGÉTICO
=============================================================================

Modelos que suportam NaN nativamente:
- HistGradientBoostingRegressor (sklearn)
- LightGBM
- XGBoost (com parâmetros específicos)
- CatBoost

Este script testa vários modelos e hiperparâmetros automaticamente.

INSTRUÇÕES:
1. Instalar dependências: pip install lightgbm xgboost catboost optuna
2. Ajustar os paths para os teus ficheiros
3. Correr o script

=============================================================================
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
import warnings
warnings.filterwarnings('ignore')

# Tentar importar bibliotecas opcionais
try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
    print("✅ LightGBM disponível")
except ImportError:
    HAS_LIGHTGBM = False
    print("⚠️ LightGBM não instalado (pip install lightgbm)")

try:
    import xgboost as xgb
    HAS_XGBOOST = True
    print("✅ XGBoost disponível")
except ImportError:
    HAS_XGBOOST = False
    print("⚠️ XGBoost não instalado (pip install xgboost)")

try:
    from catboost import CatBoostRegressor
    HAS_CATBOOST = True
    print("✅ CatBoost disponível")
except ImportError:
    HAS_CATBOOST = False
    print("⚠️ CatBoost não instalado (pip install catboost)")


def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


def erro_percentual(y_true, y_pred):
    """Erro percentual agregado (para previsão mensal)"""
    return abs(y_pred.sum() - y_true.sum()) / y_true.sum() * 100


# =============================================================================
# DEFINIÇÃO DOS MODELOS E HIPERPARÂMETROS
# =============================================================================

def get_models():
    """
    Retorna dicionário com modelos e suas variações de hiperparâmetros.
    Todos suportam NaN nativamente.
    """
    models = {}
    
    # -------------------------------------------------------------------------
    # HistGradientBoosting (sklearn) - Sempre disponível
    # -------------------------------------------------------------------------
    models["HistGB_default"] = HistGradientBoostingRegressor(
        random_state=42
    )
    
    models["HistGB_deep"] = HistGradientBoostingRegressor(
        max_depth=12,
        learning_rate=0.05,
        max_iter=500,
        l2_regularization=0.1,
        random_state=42
    )
    
    models["HistGB_shallow"] = HistGradientBoostingRegressor(
        max_depth=4,
        learning_rate=0.1,
        max_iter=300,
        l2_regularization=1.0,
        random_state=42
    )
    
    models["HistGB_tuned"] = HistGradientBoostingRegressor(
        max_depth=8,
        learning_rate=0.03,
        max_iter=500,
        min_samples_leaf=20,
        l2_regularization=0.5,
        random_state=42,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=30
    )
    
    # -------------------------------------------------------------------------
    # LightGBM - Muito rápido, suporta NaN
    # -------------------------------------------------------------------------
    if HAS_LIGHTGBM:
        models["LightGBM_default"] = lgb.LGBMRegressor(
            random_state=42,
            verbosity=-1
        )
        
        models["LightGBM_tuned"] = lgb.LGBMRegressor(
            n_estimators=500,
            max_depth=8,
            learning_rate=0.05,
            num_leaves=31,
            min_child_samples=20,
            reg_alpha=0.1,
            reg_lambda=0.1,
            random_state=42,
            verbosity=-1
        )
        
        models["LightGBM_deep"] = lgb.LGBMRegressor(
            n_estimators=1000,
            max_depth=15,
            learning_rate=0.02,
            num_leaves=63,
            min_child_samples=10,
            random_state=42,
            verbosity=-1
        )
    
    # -------------------------------------------------------------------------
    # XGBoost - Suporta NaN com tree_method apropriado
    # -------------------------------------------------------------------------
    if HAS_XGBOOST:
        models["XGBoost_default"] = xgb.XGBRegressor(
            tree_method="hist",  # Suporta NaN
            enable_categorical=False,
            random_state=42,
            verbosity=0
        )
        
        models["XGBoost_tuned"] = xgb.XGBRegressor(
            n_estimators=500,
            max_depth=8,
            learning_rate=0.05,
            tree_method="hist",
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=42,
            verbosity=0
        )
    
    # -------------------------------------------------------------------------
    # CatBoost - Excelente com NaN e features categóricas
    # -------------------------------------------------------------------------
    if HAS_CATBOOST:
        models["CatBoost_default"] = CatBoostRegressor(
            random_state=42,
            verbose=0
        )
        
        models["CatBoost_tuned"] = CatBoostRegressor(
            iterations=500,
            depth=8,
            learning_rate=0.05,
            l2_leaf_reg=3.0,
            random_state=42,
            verbose=0
        )
    
    return models


# =============================================================================
# CROSS-VALIDATION TEMPORAL
# =============================================================================

def temporal_cv_evaluate(model, X, y, n_splits=5):
    """
    Avaliação com TimeSeriesSplit (respeita ordem temporal).
    """
    tscv = TimeSeriesSplit(n_splits=n_splits)
    
    mae_scores = []
    rmse_scores = []
    erro_pct_scores = []
    
    for train_idx, val_idx in tscv.split(X):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        model.fit(X_train, y_train)
        pred = model.predict(X_val)
        
        mae_scores.append(mean_absolute_error(y_val, pred))
        rmse_scores.append(rmse(y_val, pred))
        erro_pct_scores.append(erro_percentual(y_val, pred))
    
    return {
        "MAE_mean": np.mean(mae_scores),
        "MAE_std": np.std(mae_scores),
        "RMSE_mean": np.mean(rmse_scores),
        "RMSE_std": np.std(rmse_scores),
        "ErroPct_mean": np.mean(erro_pct_scores),
        "ErroPct_std": np.std(erro_pct_scores)
    }


# =============================================================================
# MAIN - AUTOML
# =============================================================================

def run_automl(X_train, y_train, X_val, y_val, cv_splits=5):
    """
    Corre todos os modelos e retorna ranking.
    """
    models = get_models()
    results = []
    
    print("\n" + "="*70)
    print("🤖 AUTOML - TESTANDO MODELOS")
    print("="*70)
    print(f"\nModelos a testar: {len(models)}")
    print(f"Treino: {len(X_train)} registos")
    print(f"Validação: {len(X_val)} registos")
    print(f"Cross-validation: {cv_splits} splits")
    
    for name, model in models.items():
        print(f"\n{'─'*50}")
        print(f"🔄 Testando: {name}")
        
        try:
            # Cross-validation no treino
            cv_results = temporal_cv_evaluate(
                model.__class__(**model.get_params()),  # Clone fresh
                X_train, y_train, 
                n_splits=cv_splits
            )
            
            # Treinar no treino completo e avaliar na validação
            model.fit(X_train, y_train)
            pred_val = model.predict(X_val)
            
            val_mae = mean_absolute_error(y_val, pred_val)
            val_rmse = rmse(y_val, pred_val)
            val_erro_pct = erro_percentual(y_val, pred_val)
            
            results.append({
                "model": name,
                "CV_MAE": cv_results["MAE_mean"],
                "CV_MAE_std": cv_results["MAE_std"],
                "CV_RMSE": cv_results["RMSE_mean"],
                "Val_MAE": val_mae,
                "Val_RMSE": val_rmse,
                "Val_ErroPct": val_erro_pct
            })
            
            print(f"   CV MAE: {cv_results['MAE_mean']:.4f} (±{cv_results['MAE_std']:.4f})")
            print(f"   Val MAE: {val_mae:.4f} | RMSE: {val_rmse:.4f} | Erro%: {val_erro_pct:.2f}%")
            
        except Exception as e:
            print(f"   ❌ Erro: {e}")
    
    # Criar DataFrame e ordenar
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values("Val_MAE")
    
    return results_df, models


def print_results(results_df):
    """Imprime resultados formatados."""
    print("\n" + "="*70)
    print("🏆 RANKING FINAL (ordenado por Val_MAE)")
    print("="*70)
    
    print(f"\n{'Rank':<5} {'Modelo':<25} {'CV_MAE':<12} {'Val_MAE':<10} {'Val_RMSE':<10} {'Erro%':<8}")
    print("─"*75)
    
    for i, row in results_df.iterrows():
        rank = results_df.index.get_loc(i) + 1
        print(f"{rank:<5} {row['model']:<25} {row['CV_MAE']:.4f}±{row['CV_MAE_std']:.3f}  "
              f"{row['Val_MAE']:.4f}     {row['Val_RMSE']:.4f}     {row['Val_ErroPct']:.2f}%")
    
    # Melhor modelo
    best = results_df.iloc[0]
    print(f"\n🥇 MELHOR MODELO: {best['model']}")
    print(f"   MAE: {best['Val_MAE']:.4f}")
    print(f"   RMSE: {best['Val_RMSE']:.4f}")
    print(f"   Erro agregado: {best['Val_ErroPct']:.2f}%")


# =============================================================================
# SCRIPT PRINCIPAL
# =============================================================================

if __name__ == "__main__":
    
    print("="*70)
    print("🔌 AUTOML - PREVISÃO DE CONSUMO ENERGÉTICO")
    print("="*70)
    
    # =========================================================================
    # CONFIGURAÇÃO - AJUSTAR PATHS
    # =========================================================================
    
    # Opção 1: Usar dataset merged (consumo + meteorologia)
    # MERGED_PATH = "merged_hourly_weather.csv"
    
    # Opção 2: Carregar separadamente
    CONSUMPTION_PATH = "hourly.csv"  # Ajustar
    WEATHER_PATH = "weather_sceaux_2006_2010.csv"  # Ajustar
    
    TARGET = "target_kwh_hour"
    
    # =========================================================================
    # CARREGAR E PREPARAR DADOS
    # =========================================================================
    
    print("\n📂 Carregando dados...")
    
    # Se tens o merged já pronto:
    # df = pd.read_csv(MERGED_PATH, parse_dates=["Datetime"], index_col="Datetime")
    
    # Se precisas fazer merge:
    try:
        consumption = pd.read_csv(CONSUMPTION_PATH, parse_dates=["Datetime"])
        weather = pd.read_csv(WEATHER_PATH, parse_dates=["Datetime"])
        df = consumption.merge(weather, on="Datetime", how="left")
        df = df.set_index("Datetime").sort_index()
        print(f"✅ Dados carregados: {len(df)} registos")
    except FileNotFoundError:
        print("❌ Ficheiros não encontrados. A usar dados de exemplo...")
        # Fallback para o sample
        df = pd.read_csv("/mnt/project/df_sample_10000.csv", parse_dates=["Datetime"])
        df = df.set_index("Datetime").sort_index()
        print(f"⚠️ Usando dataset de amostra: {len(df)} registos")
    
    # =========================================================================
    # FEATURE ENGINEERING
    # =========================================================================
    
    print("\n🔧 Criando features...")
    
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
    
    # Lags (deixar NaN - modelos suportam)
    for lag in [1, 24, 168]:
        df[f"lag_{lag}"] = df[TARGET].shift(lag)
    
    # Meteorologia (se disponível)
    if "temperature" in df.columns:
        df["heating_degree"] = np.maximum(0, 18 - df["temperature"])
        df["cooling_degree"] = np.maximum(0, df["temperature"] - 24)
        print("   ✅ Features meteorológicas adicionadas")
    
    # Features a usar
    feature_cols = [
        "is_weekend", "hour_sin", "hour_cos", "dow_sin", "dow_cos",
        "month_sin", "month_cos", "lag_1", "lag_24", "lag_168"
    ]
    
    # Adicionar meteo se existir
    meteo_cols = ["temperature", "humidity", "precipitation", "cloud_cover", 
                  "wind_speed", "is_day", "heating_degree", "cooling_degree"]
    for col in meteo_cols:
        if col in df.columns:
            feature_cols.append(col)
    
    print(f"   Features: {len(feature_cols)}")
    
    # =========================================================================
    # SPLIT TEMPORAL
    # =========================================================================
    
    print("\n📅 Split temporal...")
    
    # Detectar período
    start = df.index.min()
    end = df.index.max()
    total_days = (end - start).days
    
    # 70% treino, 15% validação, 15% teste
    train_end = start + pd.DateOffset(days=int(total_days * 0.7))
    val_end = start + pd.DateOffset(days=int(total_days * 0.85))
    
    train = df[df.index <= train_end]
    val = df[(df.index > train_end) & (df.index <= val_end)]
    test = df[df.index > val_end]
    
    print(f"   Treino: {train.index.min().date()} → {train.index.max().date()} ({len(train)} registos)")
    print(f"   Val:    {val.index.min().date()} → {val.index.max().date()} ({len(val)} registos)")
    print(f"   Teste:  {test.index.min().date()} → {test.index.max().date()} ({len(test)} registos)")
    
    # Preparar X, y
    X_train = train[feature_cols]
    y_train = train[TARGET]
    X_val = val[feature_cols]
    y_val = val[TARGET]
    X_test = test[feature_cols]
    y_test = test[TARGET]
    
    # =========================================================================
    # CORRER AUTOML
    # =========================================================================
    
    results_df, models = run_automl(X_train, y_train, X_val, y_val, cv_splits=5)
    
    # =========================================================================
    # RESULTADOS
    # =========================================================================
    
    print_results(results_df)
    
    # =========================================================================
    # AVALIAR MELHOR MODELO NO TESTE
    # =========================================================================
    
    print("\n" + "="*70)
    print("🧪 AVALIAÇÃO FINAL NO CONJUNTO DE TESTE")
    print("="*70)
    
    best_model_name = results_df.iloc[0]["model"]
    best_model = models[best_model_name]
    
    # Retreinar no treino+validação
    X_train_full = pd.concat([X_train, X_val])
    y_train_full = pd.concat([y_train, y_val])
    
    best_model.fit(X_train_full, y_train_full)
    pred_test = best_model.predict(X_test)
    
    test_mae = mean_absolute_error(y_test, pred_test)
    test_rmse = rmse(y_test, pred_test)
    test_erro_pct = erro_percentual(y_test, pred_test)
    
    print(f"\nModelo: {best_model_name}")
    print(f"MAE (teste):  {test_mae:.4f}")
    print(f"RMSE (teste): {test_rmse:.4f}")
    print(f"Erro agregado: {test_erro_pct:.2f}%")
    
    # =========================================================================
    # GUARDAR RESULTADOS
    # =========================================================================
    
    results_df.to_csv("automl_results.csv", index=False)
    print("\n✅ Resultados guardados em automl_results.csv")
    
    # =========================================================================
    # PREVISÃO MENSAL NO TESTE
    # =========================================================================
    
    print("\n" + "="*70)
    print("📊 PREVISÃO MENSAL (TESTE)")
    print("="*70)
    
    test_df = pd.DataFrame({
        "y_true": y_test,
        "y_pred": pred_test
    }, index=y_test.index)
    
    monthly = test_df.resample("ME").sum()
    monthly["abs_error"] = (monthly["y_pred"] - monthly["y_true"]).abs()
    monthly["pct_error"] = monthly["abs_error"] / monthly["y_true"] * 100
    
    print(monthly[["y_true", "y_pred", "abs_error", "pct_error"]].round(2).to_string())
    print(f"\nMédia erro mensal: {monthly['pct_error'].mean():.2f}%")
