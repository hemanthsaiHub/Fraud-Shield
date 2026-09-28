import json, math, os, random, urllib.request
from pathlib import Path
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="FraudShield API")
M = json.loads((Path(__file__).parent / "model.json").read_text())
LABELS = {"amount": "high amount", "hour": "unusual hour", "distance_km": "far from home",
          "txn_last_hour": "many recent transactions", "new_device": "new device",
          "amount_vs_avg": "amount far above user's average", "international": "international payment"}

class Txn(BaseModel):
    amount: float; hour: int; distance_km: float; txn_last_hour: int
    new_device: int; amount_vs_avg: float; international: int

def score(t: dict):
    contrib = {}
    for i, f in enumerate(M["features"]):
        contrib[f] = M["w"][i] * (float(t[f]) - M["mean"][i]) / M["std"][i]
    z = M["b"] + sum(contrib.values())
    p = 1 / (1 + math.exp(-max(min(z, 30), -30)))
    top = sorted(((k, v) for k, v in contrib.items() if v > 0.3), key=lambda x: -x[1])[:3]
    reasons = [LABELS[k] for k, _ in top]
    return p, reasons

def explain(p, reasons, t):
    base = (f"Risk {p:.0%}. " + ("Flagged mainly due to: " + ", ".join(reasons) + "." if reasons
            else "No strong risk signals."))
    key = os.getenv("GEMINI_API_KEY")
    if not key or p < .5: return base
    try:  # optional LLM explanation (free Gemini key)
        body = json.dumps({"contents": [{"parts": [{"text":
            f"In one short sentence, explain to a bank analyst why this transaction looks fraudulent: {t}. Signals: {reasons}"}]}]}).encode()
        r = urllib.request.Request("https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key=" + key,
                                   body, {"Content-Type": "application/json"})
        return json.load(urllib.request.urlopen(r, timeout=8))["candidates"][0]["content"]["parts"][0]["text"].strip()
    except Exception:
        return base

def result(t: dict):
    p, reasons = score(t)
    level = "HIGH" if p >= .7 else "MEDIUM" if p >= .4 else "LOW"
    return {"txn": t, "risk": round(p, 4), "level": level, "reasons": reasons, "explanation": explain(p, reasons, t)}

@app.get("/api/health")
def health(): return {"status": "ok", "metrics": M["metrics"]}

@app.post("/api/predict")
def predict(t: Txn): return result(t.model_dump())

@app.get("/api/simulate")
def simulate():
    fraud = random.random() < .25
    t = dict(amount=round(random.lognormvariate(5.6 if fraud else 3.8, .9), 2),
             hour=random.choice([1, 2, 3, 4]) if fraud else random.randint(8, 22),
             distance_km=round(random.expovariate(1 / (250 if fraud else 8)), 1),
             txn_last_hour=random.randint(3, 8) if fraud else random.randint(0, 2),
             new_device=int(random.random() < (.65 if fraud else .03)),
             amount_vs_avg=round(random.lognormvariate(1.6 if fraud else 0, .6), 2),
             international=int(random.random() < (.45 if fraud else .05)))
    return result(t)

if not os.getenv("VERCEL"):  # local dev: serve the dashboard too
    from fastapi.staticfiles import StaticFiles
    app.mount("/", StaticFiles(directory=Path(__file__).parent.parent / "public", html=True), name="site")
