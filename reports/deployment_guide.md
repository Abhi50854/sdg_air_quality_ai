# 📦 Deployment Guide

## Overview

This guide provides step-by-step instructions for deploying the Air Quality Prediction Platform in various environments: local development, Docker containers, and cloud platforms.

---

## Table of Contents

1. [Local Deployment](#local-deployment)
2. [Docker Deployment](#docker-deployment)
3. [Cloud Deployment](#cloud-deployment)
4. [Monitoring & Maintenance](#monitoring--maintenance)
5. [Troubleshooting](#troubleshooting)

---

## Local Deployment

### Prerequisites
- Python 3.11
- pip
- 8GB RAM minimum
- 2GB free disk space

### Steps

1. **Clone Repository**
   ```bash
   git clone <repository-url>
   cd sdg_air_quality_ai
   ```

2. **Create Virtual Environment**
   ```bash
   python -m venv venv
   
   # Windows
   venv\Scripts\activate
   
   # Mac/Linux
   source venv/bin/activate
   ```

3. **Install Dependencies**
   ```bash
   pip install -r deployment/requirements.txt
   ```

4. **Train Model (First Time)**
   ```bash
   python src/models/train.py
   ```
   
   Expected output:
   - Data downloaded to `data/processed/`
   - Model saved to `models/saved/final_model.keras`
   - Training logs in `logs/`

5. **Run Web Application**
   ```bash
   streamlit run src/web/streamlit_app.py
   ```
   
   Access at: `http://localhost:8501`

6. **Run API (Optional)**
   ```bash
   python src/api/app.py
   ```
   
   Access at: `http://localhost:5000`

---

## Docker Deployment

### Prerequisites
- Docker Desktop installed
- Docker Compose installed

### Quick Start

1. **Build and Run**
   ```bash
   cd deployment
   docker-compose up --build
   ```

2. **Access Services**
   - **Web App**: http://localhost:8501
   - **API**: http://localhost:5000
   - **Redis**: localhost:6379

3. **Stop Services**
   ```bash
   docker-compose down
   ```

### Production Configuration

**Dockerfile Optimizations**:
```dockerfile
# Use production WSGI server for API
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "src.api.app:app"]
```

**Environment Variables** (`.env` file):
```env
FLASK_ENV=production
MODEL_PATH=models/saved/final_model.keras
ENABLE_CACHING=true
REDIS_URL=redis://redis:6379
```

**Docker Compose Production**:
```yaml
services:
  api:
    restart: always
    environment:
      - FLASK_ENV=production
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:5000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
```

---

## Streamlit Community Cloud (Easiest)

**Advantages**: Free, one-click deployment, GitHub integration

**Steps**:

1. **Push to GitHub**
   Ensure your project is on GitHub. The repository should include:
   - `src/web/streamlit_app.py`
   - `deployment/requirements.txt`
   - `models/saved/final_model.keras` (if trained)

2. **Sign in to Streamlit**
   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Sign in with GitHub

3. **Deploy App**
   - Click **"New app"**
   - Select your repository: `sdg_air_quality_ai`
   - **Main file path**: `src/web/streamlit_app.py`
   - Click **"Deploy!"**

4. **Advanced Settings (Optional)**
   If you need to set environment variables (e.g., for API keys):
   - Click "Advanced settings" before deploying
   - Add your secrets in the "Secrets" field

**Note**: Since your `requirements.txt` is in a subdirectory (`deployment/requirements.txt`), Streamlit Cloud might not find it automatically. You may need to:
- Move `deployment/requirements.txt` to the root directory OR
- Specify the path in the deployment settings if available (usually it looks in root).
**Recommendation**: Copy `deployment/requirements.txt` to the root directory for smoother deployment.

---

## Cloud Deployment

### Option 1: Google Cloud Run

**Advantages**: Serverless, auto-scaling, HTTPS included

**Steps**:

1. **Build Container**
   ```bash
   gcloud builds submit --tag gcr.io/[PROJECT-ID]/air-quality-predictor
   ```

2. **Deploy**
   ```bash
   gcloud run deploy air-quality-predictor \
     --image gcr.io/[PROJECT-ID]/air-quality-predictor \
     --platform managed \
     --region us-central1 \
     --allow-unauthenticated \
     --memory 2Gi
   ```

3. **Access**
   - URL provided in deployment output
   - Example: `https://air-quality-predictor-xxxxx.run.app`

**Cost Estimate**: ~$10-30/month for moderate traffic

---

### Option 2: AWS EC2

**Advantages**: Full control, persistent storage

**Steps**:

1. **Launch EC2 Instance**
   - AMI: Ubuntu 22.04 LTS
   - Instance type: t3.medium (2 vCPU, 4GB RAM)
   - Security Group: Open ports 80, 443, 8501

2. **SSH Into Instance**
   ```bash
   ssh -i your-key.pem ubuntu@[EC2-PUBLIC-IP]
   ```

3. **Install Dependencies**
   ```bash
   sudo apt update
   sudo apt install -y python3-pip docker.io docker-compose git
   ```

4. **Clone and Deploy**
   ```bash
   git clone <repository-url>
   cd sdg_air_quality_ai/deployment
   sudo docker-compose up -d
   ```

5. **Setup Nginx Reverse Proxy**
   ```nginx
   server {
       listen 80;
       server_name your-domain.com;
       
       location / {
           proxy_pass http://localhost:8501;
           proxy_http_version 1.1;
           proxy_set_header Upgrade $http_upgrade;
           proxy_set_header Connection 'upgrade';
       }
       
       location /api {
           proxy_pass http://localhost:5000;
       }
   }
   ```

6. **Setup SSL with Let's Encrypt**
   ```bash
   sudo apt install certbot python3-certbot-nginx
   sudo certbot --nginx -d your-domain.com
   ```

**Cost Estimate**: ~$15-40/month

---

### Option 3: Azure App Service

**Advantages**: Easy setup, integrated monitoring

**Steps**:

1. **Create Resource Group**
   ```bash
   az group create --name air-quality-rg --location eastus
   ```

2. **Create App Service Plan**
   ```bash
   az appservice plan create \
     --name air-quality-plan \
     --resource-group air-quality-rg \
     --sku B1 --is-linux
   ```

3. **Deploy Container**
   ```bash
   az webapp create \
     --resource-group air-quality-rg \
     --plan air-quality-plan \
     --name air-quality-app \
     --deployment-container-image-name [DOCKER-IMAGE]
   ```

**Cost Estimate**: ~$13-55/month

---

## Monitoring & Maintenance

### Model Retraining

**Frequency**: Monthly or when accuracy drops

**Automated Retraining Script**:
```bash
#!/bin/bash
# retrain.sh

cd /app
source venv/bin/activate

# Download new data
python src/models/train.py

# Evaluate performance
python src/models/evaluate.py

# If RMSE < 15, deploy new model
if [ $? -eq 0 ]; then
    echo "Model update successful"
    # Restart services
    docker-compose restart
else
    echo "Model update failed, keeping old model"
fi
```

**Cron Job**:
```cron
0 2 1 * * /app/retrain.sh >> /var/log/retrain.log 2>&1
```

### Performance Monitoring

**Metrics to Track**:
- Model RMSE/MAE
- API response times
- Error rates
- User count

**Tools**:
- **Prometheus + Grafana**: Metrics visualization
- **Logs**: Centralized with ELK stack
- **Uptime**: UptimeRobot or Pingdom

### Backup Strategy

**What to Backup**:
- Trained model files (`models/saved/`)
- User configuration (if stored)
- Logs (last 30 days)

**Automated Backup**:
```bash
# backup.sh
tar -czf backup-$(date +%Y%m%d).tar.gz models/ logs/
aws s3 cp backup-$(date +%Y%m%d).tar.gz s3://your-bucket/backups/
```

---

## Troubleshooting

### Common Issues

**Issue 1: "Model not found" error**
```
Solution: Run python src/models/train.py first
```

**Issue 2: API request fails with 500 error**
```
Solution: Check logs in logs/api.log
Verify model and scalers are loaded
```

**Issue 3: Docker container exits immediately**
```
Solution: Check Docker logs:
docker-compose logs api
Ensure requirements.txt is complete
```

**Issue 4: Streamlit app is slow**
```
Solution: 
- Enable caching (@st.cache_resource)
- Reduce data visualization points
- Upgrade instance RAM
```

**Issue 5: Out of memory during training**
```
Solution:
- Reduce batch size in train.py
- Use smaller lookback window
- Add swap space (Linux)
```

### Logs

**View API Logs**:
```bash
docker-compose logs -f api
```

**View Web App Logs**:
```bash
docker-compose logs -f web
```

**Model Training Logs**:
```bash
cat logs/training.log
```

### Health Checks

**API Health**:
```bash
curl http://localhost:5000/health
# Expected: {"status": "healthy", "timestamp": "..."}
```

**Model Loaded**:
```bash
curl -X POST http://localhost:5000/api/health-advisory \
  -H "Content-Type: application/json" \
  -d '{"aqi": 100}'
# Expected: JSON with recommendations
```

---

## Security Best Practices

1. **Environment Variables**: Store API keys in `.env`, never commit
2. **HTTPS**: Always use SSL in production (Let's Encrypt)
3. **Firewall**: Only open necessary ports (80, 443)
4. **Updates**: Regularly update dependencies (`pip list --outdated`)
5. **Access Control**: Restrict SSH access, use key-based auth

---

## Scaling Strategy

**Vertical Scaling** (More Resources):
- Upgrade to larger instance (4GB → 8GB RAM)
- Useful when single users hit limits

**Horizontal Scaling** (More Instances):
- Load balancer distributes traffic across multiple containers
- Useful for high user count

**Caching**:
- Redis for prediction caching (store for 1 hour)
- Reduces API calls and model inference

---

## Support

For deployment issues:
1. Check [Troubleshooting](#troubleshooting) section
2. Review logs for error messages
3. Open GitHub issue with:
   - Deployment environment (Docker/Cloud)
   - Error logs
   - Steps to reproduce

---

**Deployment Checklist**:
- [ ] Model trained and saved
- [ ] Environment variables configured
- [ ] SSL certificate installed (production)
- [ ] Monitoring setup
- [ ] Backup strategy implemented
- [ ] Health checks passing
- [ ] Documentation reviewed

**Happy deploying!** 🚀
