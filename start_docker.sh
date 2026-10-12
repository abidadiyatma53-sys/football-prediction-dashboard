#!/usr/bin/env bash
set -e

echo "=============================================================================="
echo " Launching Football Prediction Pipeline via Docker Compose"
echo "=============================================================================="
echo ""
echo "Building images and launching FastAPI (Port 8000) & Streamlit (Port 8501)..."
docker compose up -d --build

echo ""
echo "Containers started successfully!"
echo "• FastAPI Swagger Docs : http://localhost:8000/docs"
echo "• Streamlit Dashboard  : http://localhost:8501"
echo ""
