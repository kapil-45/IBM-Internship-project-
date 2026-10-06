# 🏠 House Price Prediction

A full-stack machine learning application that predicts house prices using a
**Random Forest Regressor** exposed via a **Flask REST API** and visualised
through an interactive **Streamlit dashboard**.

---

## Project Structure

```
house-price-prediction/
├── data/
│   └── house_price_regression_dataset.csv   ← dataset (7 features + target)
├── backend/
│   ├── train_model.py   ← EDA + training → saves model.pkl
│   ├── app.py           ← Flask REST API  (port 5000)
│   └── model.pkl        ← generated artefact (after training)
├── frontend/
│   └── app.py           ← Streamlit UI    (port 8501)
├── plots/               ← EDA plots generated during training
├── requirements.txt
└── README.md
```

---

## Dataset Features

| Feature | Description |
|---|---|
| `Square_Footage` | Total interior area (sq ft) |
| `Num_Bedrooms` | Number of bedrooms |
| `Num_Bathrooms` | Number of bathrooms |
| `Year_Built` | Year the house was constructed |
| `Lot_Size` | Lot area in acres |
| `Garage_Size` | Number of cars the garage fits |
| `Neighborhood_Quality` | Quality score 1–10 |
| `House_Price` | **Target** — market value in USD |

---

## Quick Start

### 1 — Install dependencies

```bash
pip install -r requirements.txt
```

### 2 — Train the model

```bash
python backend/train_model.py
```

This will:
- Perform EDA and save plots to `plots/`
- Compare Linear Regression, Gradient Boosting, and Random Forest
- Save the best model to `backend/model.pkl`

### 3 — Start the Flask API

```bash
python backend/app.py
```

API runs at **http://localhost:5000**

### 4 — Launch the Streamlit UI

Open a **second terminal**:

```bash
streamlit run frontend/app.py
```

UI opens at **http://localhost:8501**

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET`  | `/health` | Service status + model metrics |
| `GET`  | `/feature-ranges` | Min/max/mean per feature (for UI) |
| `POST` | `/predict` | Single house prediction |
| `POST` | `/batch-predict` | Multiple houses at once |

### Example — POST /predict

```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "Square_Footage": 2500,
    "Num_Bedrooms": 3,
    "Num_Bathrooms": 2,
    "Year_Built": 2005,
    "Lot_Size": 2.5,
    "Garage_Size": 1,
    "Neighborhood_Quality": 7
  }'
```

Response:
```json
{
  "predicted_price": 573420.50,
  "model": "Random Forest",
  "inputs_used": { ... },
  "metrics": { "MAE": 12345, "RMSE": 18234, "R2": 0.9876 }
}
```

---

## UI Tabs

| Tab | Content |
|-----|---------|
| **📊 Predict** | Live price prediction with sliders + sensitivity chart |
| **📈 EDA** | Distribution, scatter plots, correlation heatmap |
| **⚙️ Model Info** | Performance metrics, feature importances |
| **🗂️ Compare Houses** | Add up to 5 houses and compare predictions |

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| ML Model | scikit-learn `RandomForestRegressor` |
| Backend | Python · Flask · flask-cors |
| Frontend | Python · Streamlit · Plotly |
| Data | pandas · NumPy · seaborn · matplotlib |
