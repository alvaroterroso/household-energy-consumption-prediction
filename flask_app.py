"""

FLOW:
1. POST /upload    → gets CSV (simulates smart meter data)
2. POST /train     → trains model with csv
3. POST /predict   → preview next month
4. GET  /status    → check current status

COMO CORRER:
    pip install flask pandas numpy scikit-learn
    python app_flask_complete.py
    
    Open: http://localhost:5001

"""

from flask import Flask, request, jsonify, render_template_string
import pandas as pd
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
import io
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)

# Global system state
SYSTEM_STATE = {
    "data": None,           # DataFrame with load data
    "model": None,          # Trained model
    "last_date": None,      # Last date in data
    "next_month": None,     # Month to predict
    "is_trained": False,    # Is model trained?
    "training_info": None   # Last training info
}

# Portugal fee structure (€/kWh)
FEE = {
    "simple": {"standard": 0.1654},
    "bi_scheduled": {"vazio": 0.1023, "fora_vazio": 0.1892},
    "tri_scheduled": {"vazio": 0.1023, "cheias": 0.1654, "ponta": 0.2145}
}

# =============================================================================
# FUNÇÕES DO MODELO
# =============================================================================

def prepare_features(df, target_col="target_kwh_hour"):
    """Prepara features para treino/previsão."""
    df = df.copy()
    
    # Calendário
    df["hour"] = df.index.hour
    df["day_of_week"] = df.index.dayofweek
    df["month"] = df.index.month
    df["is_weekend"] = (df.index.dayofweek >= 5).astype(int)
    
    # Encoding cíclico
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["dow_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    
    # Lags longos
    df["lag_720"] = df[target_col].shift(720)
    df["lag_8760"] = df[target_col].shift(8760)
    

    return df


def get_feature_cols(df):
    """Retorna features disponíveis."""
    base = ["is_weekend", "hour_sin", "hour_cos", "dow_sin", "dow_cos",
            "month_sin", "month_cos", "lag_720", "lag_8760"]
    meteo = ["temperature", "humidity", "heating_degree", "cooling_degree",
            "precipitation","cloud_cover","wind_speed","is_day"]
    
    features = [f for f in base if f in df.columns]
    features += [f for f in meteo if f in df.columns]
    return features


def train_model(df, target_col="target_kwh_hour"):
    """Treina o modelo com os dados fornecidos."""
    
    # Preparar features
    df_prep = prepare_features(df, target_col)
    feature_cols = get_feature_cols(df_prep)
    
    # Filtrar dados válidos (com lag_720)
    train_data = df_prep.dropna(subset=[target_col, "lag_720"])
    
    if len(train_data) < 500:
        return None, f"Dados insuficientes: {len(train_data)} registos (mínimo: 500)"
    
    X = train_data[feature_cols]
    y = train_data[target_col]
    
    # Treinar
    model = HistGradientBoostingRegressor(
        max_depth=8,
        learning_rate=0.05,
        max_iter=300,
        random_state=42
    )
    model.fit(X, y)
    
    return model, feature_cols


def predict_month(model, feature_cols, df, target_col="target_kwh_hour"):
    """Prevê o próximo mês completo."""
    
    # Próximo mês após os dados
    last_date = df.index.max()
    next_month_start = (last_date + pd.Timedelta(hours=1)).replace(day=1, hour=0, minute=0, second=0)
    if next_month_start <= last_date:
        next_month_start = next_month_start + pd.DateOffset(months=1)
    next_month_end = next_month_start + pd.DateOffset(months=1) - pd.Timedelta(hours=1)
    
    # Gerar horas do mês
    forecast_hours = pd.date_range(next_month_start, next_month_end, freq='h')
    forecast_df = pd.DataFrame(index=forecast_hours)
    
    # Features de calendário
    forecast_df["hour"] = forecast_df.index.hour
    forecast_df["day_of_week"] = forecast_df.index.dayofweek
    forecast_df["month"] = forecast_df.index.month
    forecast_df["is_weekend"] = (forecast_df.index.dayofweek >= 5).astype(int)
    
    forecast_df["hour_sin"] = np.sin(2 * np.pi * forecast_df["hour"] / 24)
    forecast_df["hour_cos"] = np.cos(2 * np.pi * forecast_df["hour"] / 24)
    forecast_df["dow_sin"] = np.sin(2 * np.pi * forecast_df["day_of_week"] / 7)
    forecast_df["dow_cos"] = np.cos(2 * np.pi * forecast_df["day_of_week"] / 7)
    forecast_df["month_sin"] = np.sin(2 * np.pi * forecast_df["month"] / 12)
    forecast_df["month_cos"] = np.cos(2 * np.pi * forecast_df["month"] / 12)
    
    # Lags (do histórico)
    for hour in forecast_hours:
        for lag_name, lag_hours in [("lag_720", 720), ("lag_8760", 8760)]:
            lag_time = hour - pd.Timedelta(hours=lag_hours)
            if lag_time in df.index:
                forecast_df.loc[hour, lag_name] = df.loc[lag_time, target_col]
            else:
                forecast_df.loc[hour, lag_name] = df[target_col].mean()
    
    # Meteorologia (usar ano anterior como proxy)
    if "temperature" in df.columns:
        for hour in forecast_hours:
            year_ago = hour - pd.DateOffset(years=1)
            if year_ago in df.index:
                for col in ["temperature", "humidity"]:
                    if col in df.columns:
                        forecast_df.loc[hour, col] = df.loc[year_ago, col]
        
        for col in ["temperature", "humidity"]:
            if col in forecast_df.columns:
                forecast_df[col] = forecast_df[col].fillna(df[col].mean() if col in df.columns else 18)
        
        if "temperature" in forecast_df.columns:
            forecast_df["heating_degree"] = np.maximum(0, 18 - forecast_df["temperature"])
            forecast_df["cooling_degree"] = np.maximum(0, forecast_df["temperature"] - 24)
    
    # Garantir todas as features
    for col in feature_cols:
        if col not in forecast_df.columns:
            forecast_df[col] = 0
    
    # Prever
    X = forecast_df[feature_cols].fillna(0)
    predictions = model.predict(X)
    predictions = np.maximum(0, predictions)
    
    forecast_df["predicted_kwh"] = predictions
    
    return forecast_df, next_month_start.strftime("%Y-%m")


def calculate_cost(forecast_df, tarifa="simple"):
    """Calcula custo total."""
    total = 0
    precos = FEE.get(tarifa, FEE["simple"])
    
    for idx, row in forecast_df.iterrows():
        h = idx.hour
        kwh = row["predicted_kwh"]
        
        if tarifa == "simple":
            total += kwh * precos["standard"]
        elif tarifa == "bi_scheduled":
            if h >= 22 or h < 8:
                total += kwh * precos["vazio"]
            else:
                total += kwh * precos["fora_vazio"]
        else:  # tri_scheduled
            if h >= 22 or h < 8:
                total += kwh * precos["vazio"]
            elif (8 <= h < 10) or (18 <= h < 21):
                total += kwh * precos["cheias"]
            else:
                total += kwh * precos["ponta"]
    
    return total



HTML_TEMPLATE = '''
<!DOCTYPE html>
<html lang="pt">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Energy Predictor</title>
</head>
<body>
    <div class="container">
        <h1>Energy Predictor</h1>
        
        <!-- Flow -->
        <div class="flow">
            <div class="flow-step" id="step1">1. Upload</div>
            <div class="flow-step" id="step2">2. Train</div>
            <div class="flow-step" id="step3">3. Predict</div>
        </div>
        
        <!-- STEP 1: Upload -->
        <div class="card">
            <h2> Step 1: Upload Data from Smart Meter</h2>
        
            
            <input type="file" id="csvFile" accept=".csv">
            <button class="btn-primary" onclick="uploadData()">Load Data</button>
            
            <div class="status-box" id="uploadStatus" style="display:none;">
                <div class="status-item">
                    <span class="status-label">Ficheiro:</span>
                    <span class="status-value" id="fileName">-</span>
                </div>
                <div class="status-item">
                    <span class="status-label">Registos:</span>
                    <span class="status-value" id="numRecords">-</span>
                </div>
                <div class="status-item">
                    <span class="status-label">Período:</span>
                    <span class="status-value" id="dataPeriod">-</span>
                </div>
            </div>
        </div>
        
        <!-- STEP 2: Treinar -->
        <div class="card">
            <h2>Step 2: Train Model</h2>
            
            <button class="btn-success" id="trainBtn" onclick="trainModel()" disabled>
                Press to Train
            </button>
            
            <div class="loading" id="trainLoading">
                <div class="spinner"></div>
                <p>Loading...</p>
            </div>
            
            <div class="status-box" id="trainStatus" style="display:none;">
                <div class="status-item">
                    <span class="status-label">State:</span>
                    <span class="status-value" id="trainState">-</span>
                </div>
                <div class="status-item">
                    <span class="status-label">Training data:</span>
                    <span class="status-value" id="trainRecords">-</span>
                </div>
                <div class="status-item">
                    <span class="status-label">Last Month:</span>
                    <span class="status-value" id="lastMonth">-</span>
                </div>
                <div class="status-item">
                    <span class="status-label">Next Month (predicting):</span>
                    <span class="status-value" id="nextMonth">-</span>
                </div>
            </div>
        </div>
        
        <!-- Step 3: Prediction -->
        <div class="card">
            <h2> Step 3: Predicting Next Month Consumption</h2>
            
            <div style="margin-bottom: 15px;">
                <label>Consumption fee:</label>
                <select id="tarifa">
                    <option value="simple">Standard (0.1654 €/kWh)</option>
                    <option value="bi_scheduled">Bi-Scheduled</option>
                    <option value="tri_scheduled">Tri-Scheduled</option>
                </select>
            </div>
            
            <button class="btn-warning" id="predictBtn" onclick="predict()" disabled>
                Predict Next Month
            </button>
            
            <div class="loading" id="predictLoading">
                <div class="spinner"></div>
                <p>Loading...</p>
            </div>
            
            <div class="result-box" id="resultBox">
                <h3 id="resultMonth">Prediction: -</h3>
                
                <div class="metrics">
                    <div class="metric">
                    	<div class="result-label">kWh total</div>
                        <div class="big-number" id="resultKwh">-</div>
                    </div>
                    <div class="metric">
                    	<div class="result-label">€ cost</div>
                        <div class="big-number" id="resultCost">-</div>
                    </div>
                    <div class="metric">
                        <div class="result-label">kWh/daily average</div>
                        <div class="big-number" id="resultAvg">-</div>
                    </div>
                </div>
            </div>
        </div>
    </div>
    
    <script>
        function updateFlowStep(step) {
            document.querySelectorAll('.flow-step').forEach((el, i) => {
                el.classList.remove('active', 'done');
                if (i + 1 < step) el.classList.add('done');
                if (i + 1 === step) el.classList.add('active');
            });
        }
        
        async function uploadData() {
            const fileInput = document.getElementById('csvFile');
            const file = fileInput.files[0];
            
            if (!file) {
                alert('Please use a CSV file to upload data.');
                return;
            }
            
            const formData = new FormData();
            formData.append('file', file);
            
            try {
                const response = await fetch('/upload', {
                    method: 'POST',
                    body: formData
                });
                
                const result = await response.json();
                
                if (result.success) {
                    document.getElementById('uploadStatus').style.display = 'block';
                    document.getElementById('fileName').textContent = file.name;
                    document.getElementById('numRecords').textContent = result.num_records.toLocaleString();
                    document.getElementById('dataPeriod').textContent = result.period;
                    
                    document.getElementById('trainBtn').disabled = false;
                    updateFlowStep(2);
                } else {
                    alert('Error: ' + result.error);
                }
            } catch (error) {
                alert('Conection error: ' + error.message);
            }
        }
        
        async function trainModel() {
            document.getElementById('trainLoading').style.display = 'block';
            document.getElementById('trainBtn').disabled = true;
            
            try {
                const response = await fetch('/train', { method: 'POST' });
                const result = await response.json();
                
                document.getElementById('trainLoading').style.display = 'none';
                
                if (result.success) {
                    document.getElementById('trainStatus').style.display = 'block';
                    document.getElementById('trainState').textContent = 'Trained';
                    document.getElementById('trainRecords').textContent = result.train_records.toLocaleString();
                    document.getElementById('lastMonth').textContent = result.last_month;
                    document.getElementById('nextMonth').textContent = result.next_month;
                    
                    document.getElementById('predictBtn').disabled = false;
                    updateFlowStep(3);
                } else {
                    document.getElementById('trainBtn').disabled = false;
                    alert('Erro: ' + result.error);
                }
            } catch (error) {
                document.getElementById('trainLoading').style.display = 'none';
                document.getElementById('trainBtn').disabled = false;
                alert('Erro: ' + error.message);
            }
        }
        
        async function predict() {
            const tarifa = document.getElementById('tarifa').value;
            document.getElementById('predictLoading').style.display = 'block';
            document.getElementById('predictBtn').disabled = true;
            
            try {
                const response = await fetch('/predict', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ tarifa: tarifa })
                });
                
                const result = await response.json();
                
                document.getElementById('predictLoading').style.display = 'none';
                document.getElementById('predictBtn').disabled = false;
                
                if (result.success) {
                    document.getElementById('resultBox').style.display = 'block';
                    document.getElementById('resultBox').classList.remove('error');
                    document.getElementById('resultMonth').textContent = 'Prevision for: ' + result.month;
                    document.getElementById('resultKwh').textContent = result.total_kwh.toFixed(0);
                    document.getElementById('resultCost').textContent = result.total_cost.toFixed(2) + ' €';
                    document.getElementById('resultAvg').textContent = result.daily_avg.toFixed(1);
                } else {
                    document.getElementById('resultBox').style.display = 'block';
                    document.getElementById('resultBox').classList.add('error');
                    document.getElementById('resultMonth').textContent = 'Erro: ' + result.error;
                }
            } catch (error) {
                document.getElementById('predictLoading').style.display = 'none';
                document.getElementById('predictBtn').disabled = false;
                alert('Erro: ' + error.message);
            }
        }
        
        // Inicializar
        updateFlowStep(1);
    </script>
</body>
</html>
'''


# ENDPOINTS

@app.route('/')
def home():
    return render_template_string(HTML_TEMPLATE)


@app.route('/upload', methods=['POST'])
def upload_data():
    """
    Gets CSV file upload to simulate smart meter data reception.
    """
    global SYSTEM_STATE
    
    try:
        if 'file' not in request.files:
            return jsonify({"success": False, "error": "No file sent"})
        
        file = request.files['file']
        
        # Read CSV
        df = pd.read_csv(io.StringIO(file.read().decode('utf-8')), parse_dates=["Datetime"])
        df = df.set_index("Datetime").sort_index()
        
        # Save state
        SYSTEM_STATE["data"] = df
        SYSTEM_STATE["last_date"] = df.index.max()
        SYSTEM_STATE["is_trained"] = False
        SYSTEM_STATE["model"] = None
        
        return jsonify({
            "success": True,
            "num_records": len(df),
            "period": f"{df.index.min().date()} → {df.index.max().date()}",
            "last_date": str(df.index.max().date())
        })
        
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})


@app.route('/train', methods=['POST'])
def train():
    """
   Trains model on demand with uploaded data.
    """
    global SYSTEM_STATE
    
    if SYSTEM_STATE["data"] is None:
        return jsonify({"success": False, "error": "No file loaded, upload first."})
    
    try:
        df = SYSTEM_STATE["data"]
        
        model, feature_cols = train_model(df, "target_kwh_hour")
        
        if model is None:
            return jsonify({"success": False, "error": feature_cols})


        SYSTEM_STATE["model"] = model
        SYSTEM_STATE["feature_cols"] = feature_cols
        SYSTEM_STATE["is_trained"] = True
        
        
        last_date = df.index.max()
        next_month = (last_date + pd.Timedelta(hours=1)).replace(day=1)
        if next_month <= last_date:
            next_month = next_month + pd.DateOffset(months=1)
        
        SYSTEM_STATE["next_month"] = next_month.strftime("%Y-%m")
        
        return jsonify({
            "success": True,
            "train_records": len(df.dropna(subset=["target_kwh_hour"])),
            "last_month": last_date.strftime("%Y-%m"),
            "next_month": SYSTEM_STATE["next_month"],
            "features_used": len(feature_cols)
        })
        
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})


@app.route('/predict', methods=['POST'])
def predict():
    """
    Predicts next month from the dataset
    """
    global SYSTEM_STATE
    
    if not SYSTEM_STATE["is_trained"]:
        return jsonify({"success": False, "error": "Modelo não treinado. Treina primeiro."})
    
    try:
        data = request.json or {}
        tarifa = data.get("tarifa", "simples")
        
        df = SYSTEM_STATE["data"]
        model = SYSTEM_STATE["model"]
        feature_cols = SYSTEM_STATE["feature_cols"]
        
        # Prediction
        forecast_df, month = predict_month(model, feature_cols, df, "target_kwh_hour")
        
        # Calculate cost
        total_kwh = forecast_df["predicted_kwh"].sum()
        total_cost = calculate_cost(forecast_df, tarifa)
        
        # Daily average
        days = len(forecast_df) / 24
        daily_avg = total_kwh / days
        
        return jsonify({
            "success": True,
            "month": month,
            "total_kwh": float(total_kwh),
            "total_cost": float(total_cost),
            "daily_avg": float(daily_avg),
            "tarifa": tarifa,
            "hours_predicted": len(forecast_df)
        })
        
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})


@app.route('/status', methods=['GET'])
def status():
    """
    Returns current system status.
    """
    return jsonify({
        "data_loaded": SYSTEM_STATE["data"] is not None,
        "num_records": len(SYSTEM_STATE["data"]) if SYSTEM_STATE["data"] is not None else 0,
        "last_date": str(SYSTEM_STATE["last_date"]) if SYSTEM_STATE["last_date"] else None,
        "is_trained": SYSTEM_STATE["is_trained"],
        "next_month": SYSTEM_STATE["next_month"]
    })

if __name__ == '__main__':
    """
    Run Flask app
    """
    print("Energy Predictor - Flask API")
    print()
    print("Server: http://localhost:5001")
    print()
    print("Use Flow:")
    print("   1. Open http://localhost:5001 in the browser")
    print("   2. Upload CSV (data until desired month)")
    print("   3. Press 'Train Model'")
    print("   4. Press 'Preview Next Month'")
    print()
    
    app.run(debug=True, host='0.0.0.0', port=5001)
