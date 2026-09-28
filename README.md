# FraudShield – Real-Time Fraud Detection with Explainable AI

Scores transactions in real time, flags fraud, and explains *why* (per-feature contributions, plus an optional
LLM sentence via a free Gemini key). Includes a live dashboard with a transaction simulator.

**Stack:** Python, FastAPI, NumPy (training), vanilla JS dashboard, Vercel (deployment).

## Run locally
```
python -m venv venv
venv\Scripts\activate          (Windows)   |   source venv/bin/activate   (Mac/Linux)
pip install -r requirements.txt
uvicorn api.index:app --reload
```
Open http://localhost:8000 (dashboard) and http://localhost:8000/docs (API docs).

## Retrain the model (optional)
```
pip install -r requirements-train.txt
python train.py
```
Uses synthetic data by default. To use real data (e.g. Kaggle fraud dataset), make a CSV with columns
`amount,hour,distance_km,txn_last_hour,new_device,amount_vs_avg,international,is_fraud` and run `python train.py data.csv`.
Note: synthetic data gives near-perfect scores; report real-dataset results in your final report.

## Optional: LLM explanations
Get a free key at aistudio.google.com, then set `GEMINI_API_KEY` (in a `.env`/terminal, or in Vercel > Settings > Environment Variables).

## Deploy to Vercel
```
npm i -g vercel
vercel login
vercel --prod
```
Or import the GitHub repo at vercel.com/new (no build settings needed).

## API
- `GET /api/health` – model metrics
- `GET /api/simulate` – random scored transaction
- `POST /api/predict` – score your own transaction (JSON with the 7 features)
