# Quick Start Script for Windows
# Runs the complete training and deployment pipeline

Write-Host "========================================" -ForegroundColor Cyan
Write-Host " AIR QUALITY PREDICTOR - QUICK START" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Step 1: Activate virtual environment
Write-Host "[1/4] Activating virtual environment..." -ForegroundColor Yellow
& "venv\Scripts\Activate.ps1"

# Step 2: Install dependencies
Write-Host "[2/4] Installing dependencies..." -ForegroundColor Yellow
pip install -r deployment\requirements.txt

# Step 3: Train model
Write-Host "[3/4] Training AI model (this may take 5-10 minutes)..." -ForegroundColor Yellow
python src\models\train.py

# Step 4: Launch web app
Write-Host "[4/4] Launching web application..." -ForegroundColor Green
Write-Host ""
Write-Host "✓ Setup complete! Opening browser..." -ForegroundColor Green
Write-Host ""
Write-Host "Access the application at: http://localhost:8501" -ForegroundColor Cyan
Write-Host "Press Ctrl+C to stop the server" -ForegroundColor Gray
Write-Host ""

streamlit run src\web\streamlit_app.py
