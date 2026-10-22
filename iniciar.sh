#!/usr/bin/env bash
# Abre o sistema no navegador (http://localhost:8501). Funciona offline.
cd "$(dirname "$0")"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

# banco de dados: MongoDB local no Docker, com os dados em dados/mongo
if ! docker start formularios-mongo >/dev/null 2>&1; then
  echo "Criando o banco MongoDB local (só na primeira vez)..."
  mkdir -p dados/mongo
  docker run -d --name formularios-mongo --restart unless-stopped \
    -p 127.0.0.1:27017:27017 -v "$PWD/dados/mongo:/data/db" mongo:8.0.4 >/dev/null || {
    echo "Não consegui iniciar o MongoDB. O Docker está instalado e rodando?"; exit 1; }
fi

exec .venv/bin/streamlit run app.py --browser.gatherUsageStats false --server.headless true
