"""
=============================================================================
🔪 DATASET CUTTER - Preparar CSV para Upload
=============================================================================

Corta o dataset até ao mês que escolheres.
O ficheiro CSV gerado pode ser usado no app para prever o mês seguinte.

COMO FUNCIONA:
1. Carrega o dataset completo (merged_with_weather.csv)
2. Tu escolhes até que mês queres cortar
3. Gera CSV pronto para upload no Streamlit/Flask

NOTA SOBRE METEOROLOGIA:
- O CSV incluirá dados meteorológicos HISTÓRICOS
- O app usa meteorologia do ANO ANTERIOR como proxy
- Sem API integrada, a previsão baseia-se em:
  ✅ Padrões de calendário (hora, dia, mês, fim-de-semana)
  ✅ Lags longos (consumo de há 1 mês, 2 semanas, 1 ano)
  ✅ Meteorologia do ano anterior (proxy)
  
=============================================================================
"""

import pandas as pd
import os


def list_available_months(df):
    """Lista os meses disponíveis no dataset."""
    months = df.index.to_period('M').unique().sort_values()
    return months


def cut_dataset(df, year, month):
    """
    Corta o dataset até ao final do mês especificado.
    """
    # Criar data de corte (último momento do mês)
    cut_date = pd.Timestamp(year=year, month=month, day=1)
    cut_end = cut_date + pd.DateOffset(months=1) - pd.Timedelta(seconds=1)
    
    # Cortar
    df_cut = df[df.index <= cut_end].copy()
    
    return df_cut


def main():
    print("="*60)
    print("🔪 DATASET CUTTER - Preparar CSV para Previsão")
    print("="*60)
    
    # Ficheiro de entrada
    input_file = input("\n📂 Ficheiro de entrada (default: merged_with_weather.csv): ").strip()
    if not input_file:
        input_file = "merged_with_weather.csv"
    
    # Carregar dataset
    try:
        print(f"\n📊 A carregar {input_file}...")
        df = pd.read_csv(input_file, parse_dates=["Datetime"])
        df = df.set_index("Datetime").sort_index()
        print(f"   ✅ {len(df):,} registos carregados")
        print(f"   📅 Período: {df.index.min().date()} → {df.index.max().date()}")
    except FileNotFoundError:
        print(f"   ❌ Ficheiro não encontrado: {input_file}")
        return
    except Exception as e:
        print(f"   ❌ Erro ao carregar: {e}")
        return
    
    # Mostrar meses disponíveis
    months = list_available_months(df)
    print(f"\n📅 Meses disponíveis no dataset:")
    print("   ", end="")
    for i, m in enumerate(months):
        print(f"{m}", end="  ")
        if (i + 1) % 6 == 0:
            print("\n   ", end="")
    print()
    
    # Pedir mês de corte
    print("\n" + "-"*60)
    print("Escolhe até que mês queres os dados.")
    print("O app vai prever o MÊS SEGUINTE ao que escolheres.")
    print("-"*60)
    
    try:
        year = int(input("\n📆 Ano de corte (ex: 2010): "))
        month = int(input("📆 Mês de corte (1-12, ex: 7 para Julho): "))
        
        if month < 1 or month > 12:
            print("❌ Mês inválido!")
            return
            
    except ValueError:
        print("❌ Entrada inválida!")
        return
    
    # Verificar se o mês existe
    cut_period = pd.Period(year=year, month=month, freq='M')
    if cut_period not in months.values:
        print(f"❌ O mês {year}-{month:02d} não existe no dataset!")
        return
    
    # Cortar dataset
    df_cut = cut_dataset(df, year, month)
    
    # Mostrar info
    next_month = pd.Timestamp(year=year, month=month, day=1) + pd.DateOffset(months=1)
    
    print(f"\n✂️  Dataset cortado:")
    print(f"   📊 Registos: {len(df_cut):,}")
    print(f"   📅 De: {df_cut.index.min().date()}")
    print(f"   📅 Até: {df_cut.index.max().date()}")
    print(f"\n   🔮 O app vai prever: {next_month.strftime('%B %Y')}")
    
    # Guardar
    output_file = f"dataset_until_{year}_{month:02d}.csv"
    
    # Reset index para ter Datetime como coluna (necessário para o app)
    df_cut = df_cut.reset_index()
    df_cut.to_csv("test-datasets/" + output_file, index=False)
    
    print(f"\n✅ Ficheiro guardado: {output_file}")
    print(f"   📤 Usa este ficheiro no Streamlit/Flask para prever {next_month.strftime('%B %Y')}")
    
    # Resumo do que vai acontecer
    print("\n" + "="*60)
    print("📋 O QUE VAI ACONTECER NO APP:")
    print("="*60)
    print(f"""
    1. Upload do ficheiro: {output_file}
    
    2. Modelo treina com dados até {year}-{month:02d}
    
    3. Para prever {next_month.strftime('%Y-%m')}, o modelo usa:
       ✅ Calendário: hora, dia da semana, mês, fim-de-semana
       ✅ lag_720: consumo de {(next_month - pd.DateOffset(days=30)).strftime('%Y-%m')} (há ~1 mês)
       ✅ lag_336: consumo de há ~2 semanas
       ✅ lag_8760: consumo de {(next_month - pd.DateOffset(years=1)).strftime('%Y-%m')} (ano passado)
       ⚠️  Meteorologia: usa dados de {(next_month - pd.DateOffset(years=1)).strftime('%Y-%m')} como proxy
    
    4. Resultado: previsão de consumo + custo em €
    """)
    
    # Verificar se há dados reais para comparação
    real_month_start = next_month
    real_month_end = next_month + pd.DateOffset(months=1) - pd.Timedelta(hours=1)
    
    # Recarregar dataset original para verificar
    df_original = pd.read_csv(input_file, parse_dates=["Datetime"])
    df_original = df_original.set_index("Datetime").sort_index()
    
    real_data = df_original[(df_original.index >= real_month_start) & 
                            (df_original.index <= real_month_end)]
    
    if len(real_data) > 0:
        real_total = real_data["target_kwh_hour"].sum()
        print(f"📊 NOTA: O dataset original tem dados reais de {next_month.strftime('%Y-%m')}:")
        print(f"   Consumo real: {real_total:.0f} kWh")
        print(f"   Podes comparar com a previsão do app!")
    else:
        print(f"⚠️  Não há dados reais de {next_month.strftime('%Y-%m')} para comparar.")


if __name__ == "__main__":
    main()
