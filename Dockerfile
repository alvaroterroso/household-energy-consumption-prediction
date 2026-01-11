# =============================================================================
# 🐳 Dockerfile - Energy Predictor (Flask com Treino)
# =============================================================================
#
# Este container permite:
# - Upload de dados (CSV)
# - Treino do modelo
# - Previsão do próximo mês
#
# BUILD:
#   docker build -t energy-predictor .
#
# RUN:
#   docker run -d --name energy-app -p 5000:5000 energy-predictor
#
# ACEDER:
#   http://localhost:5000
#
# =============================================================================

FROM python:3.11-slim

# Metadata
LABEL maintainer="IPCA MAAI 2025/2026"
LABEL description="Household Energy Consumption Predictor"
LABEL version="1.0"

# Set working directory
WORKDIR /app

# Copy requirements first (Docker cache optimization)
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements_flask.txt

# Copy application
COPY app_flask_complete.py .

# Expose port
EXPOSE 5000

# Run application
CMD ["python", "app_flask_complete.py"]
