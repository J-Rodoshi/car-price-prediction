# Car Price Prediction System

This project predicts the selling price of used cars based on vehicle specifications and listing details. It includes a Jupyter notebook for data preparation, model training, and evaluation, plus a Dockerized Dash web application for real‑time predictions.

## Notebook
- `Car_Price_Prediction.ipynb`: Contains all EDA, preprocessing, model selection, and inference. The final XGBoost model, scaler, and column list are saved as `.pkl` files.

## Web App
The web app is built with Dash and deployed via Docker.

### Run locally
1. Install dependencies: `pip install -r app/requirements.txt`
2. Place the trained model files (`xgb_car_price_model.pkl`, `scaler.pkl`, `model_columns.pkl`) inside `app/code/`.
3. Run: `python app/code/app.py`
4. Open `http://localhost:8050`.

### Run with Docker
1. Ensure Docker is installed.
2. From the root directory, run:
   ```bash
   docker compose up --build