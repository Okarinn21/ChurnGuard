# ChurnGuard — Bank Customer Churn Prediction System

A machine learning system for predicting bank customer churn using **XGBoost** + **FastAPI** + **HTML Frontend**.

**Dataset:** 165,034 customers · 10 features · 21.2% real churn rate

---

## 📁 Project Structure

```
BANK-CHURN/
│
├── apps/
│   ├── api/
│   │   ├── __pycache__/
│   │   ├── __init__.py
│   │   ├── main.py               # FastAPI app — routes & middleware
│   │   ├── model_service.py      # Model loading, preprocessing, prediction logic
│   │   └── schemas.py            # Pydantic request/response schemas
│   ├── __init__.py
│   └── index.html                # Frontend dashboard (no install needed)
│
├── dataset/
│   ├── train.csv                 # Training data (165,034 rows)
│   ├── test.csv                  # Test data for submission
│   ├── sample.csv                # Sample input for batch prediction
│   ├── sample_submission.csv     # Submission format reference
│   └── submission_rf.csv         # Random Forest submission output
│
├── models/
│   ├── xgboost.pkl               # Active model (XGBoost)
│   ├── le_geography.pkl          # Label Encoder — Geography
│   ├── le_gender.pkl             # Label Encoder — Gender
│   ├── scaler.pkl                
│   ├── meta.json                 # Metadata: features, best_threshold, val_metrics
│   ├── eda_overview.png          # EDA visualization output
│   └── model_comparison.png     # Model comparison chart output
├── .gitignore
├── .venv/                        # Python virtual environment
├── model_evaluation.ipynb        # EDA + Training + Evaluation notebook
├── run.py                        # Server entry point
├── requirements.txt              # Python dependencies
└── README.md
```

---

## ⚙️ Requirements

- Python **3.9+**
- pip
- Any modern web browser (Chrome / Firefox / Edge)

---

## 🚀 Installation & Running

### Step 1 — Create virtual environment (recommended)

```bash
python -m venv .venv

# Activate — Windows
.venv\Scripts\activate

# Activate — macOS / Linux
source .venv/bin/activate
```

### Step 2 — Install dependencies

```bash
pip install -r requirements.txt
```

### Step 3 — Run the FastAPI backend

```bash
python run.py
```

Or using uvicorn directly:

```bash
uvicorn apps.api.main:app --reload --host 0.0.0.0 --port 8000
```

The server will start at `http://localhost:8000`

Verify the server is running:

```bash
curl http://localhost:8000/health
```

### Step 4 — Open the Frontend

Open `apps/index.html` in your browser:

```bash
# macOS
open apps/index.html

# Windows
start apps/index.html

# Linux
xdg-open apps/index.html
```

> **Note:** The frontend works even without a running API — it uses demo data as fallback. When the API is running, all data is live from the real model.

### Step 5 — Explore API Docs (optional)

| URL | Description |
|-----|-------------|
| http://localhost:8000/docs | Swagger UI — interactive API testing |
| http://localhost:8000/redoc | ReDoc — clean API documentation |

---

## 📡 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | Root — connection check |
| `GET` | `/health` | Health check + model status |
| `GET` | `/model/info` | Information about the loaded model |
| `POST` | `/predict` | Predict churn for a single customer |
| `POST` | `/predict/batch` | Upload CSV → batch prediction results (JSON) |
| `POST` | `/predict/batch/download` | Upload CSV → download result as CSV file |
| `GET` | `/stats/dataset` | Descriptive statistics of the training dataset |
| `GET` | `/stats/feature-importance` | Feature importance scores |


## 🛠️ Troubleshooting

**Error: `Model not loaded` (503)**
```bash
# Check that model files exist
ls models/
# Required: xgboost.pkl, le_geography.pkl, le_gender.pkl, meta.json
# If missing, re-run the notebook to retrain and save the model
```

**CORS error when frontend calls the API**
```bash
# main.py already sets allow_origins=["*"]
# If the issue persists, serve the frontend via a local server instead of opening the file directly:
python -m http.server 5500 --directory apps
# Then open http://localhost:5500
```

**Error: `Missing columns` when uploading CSV**
```
Check column names in your CSV — they are case-sensitive.
Correct:   NumOfProducts
Incorrect: numofproducts / Num_Of_Products / num_of_products
```

**Port 8000 already in use**
```bash
# Use a different port
uvicorn apps.api.main:app --reload --port 8001
```