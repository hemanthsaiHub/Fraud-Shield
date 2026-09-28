"""Train a fraud model (numpy only) on synthetic data and save weights to api/model.json.
Optional: python train.py path/to/your.csv  (needs columns listed in FEATURES + 'is_fraud')"""
import json, sys, numpy as np
FEATURES = ["amount", "hour", "distance_km", "txn_last_hour", "new_device", "amount_vs_avg", "international"]
rng = np.random.default_rng(42)

def synth(n=60000, fraud_rate=0.03):
    y = (rng.random(n) < fraud_rate).astype(float); f = y == 1; L = ~f
    X = np.zeros((n, 7))
    X[L, 0] = rng.lognormal(3.8, 0.9, L.sum());  X[f, 0] = rng.lognormal(5.6, 1.0, f.sum())
    X[L, 1] = np.clip(rng.normal(15, 4, L.sum()), 0, 23).astype(int)
    X[f, 1] = np.where(rng.random(f.sum()) < .6, rng.integers(0, 5, f.sum()), rng.integers(0, 24, f.sum()))
    X[L, 2] = rng.exponential(8, L.sum());       X[f, 2] = rng.exponential(250, f.sum())
    X[L, 3] = rng.poisson(0.8, L.sum());         X[f, 3] = rng.poisson(4, f.sum())
    X[L, 4] = rng.random(L.sum()) < .03;         X[f, 4] = rng.random(f.sum()) < .65
    X[L, 5] = rng.lognormal(0, .4, L.sum());     X[f, 5] = rng.lognormal(1.6, .7, f.sum())
    X[L, 6] = rng.random(L.sum()) < .05;         X[f, 6] = rng.random(f.sum()) < .45
    return X, y

def auc(y, s):
    o = np.argsort(s); r = np.empty(len(s)); r[o] = np.arange(1, len(s) + 1)
    p = y.sum(); return (r[y == 1].sum() - p * (p + 1) / 2) / (p * (len(y) - p))

if len(sys.argv) > 1:
    d = np.genfromtxt(sys.argv[1], delimiter=",", names=True)
    X = np.column_stack([d[c] for c in FEATURES]); y = d["is_fraud"]
else:
    X, y = synth()
idx = rng.permutation(len(y)); cut = int(.8 * len(y)); tr, te = idx[:cut], idx[cut:]
mean, std = X[tr].mean(0), X[tr].std(0) + 1e-9
Z = (X - mean) / std
w, b = np.zeros(7), 0.0
cw = np.where(y[tr] == 1, (y[tr] == 0).sum() / max(1, (y[tr] == 1).sum()) ** 1, 1.0) ** 0.5
for _ in range(1500):
    p = 1 / (1 + np.exp(-(Z[tr] @ w + b))); g = (p - y[tr]) * cw
    w -= 0.1 * (Z[tr].T @ g / len(tr)); b -= 0.1 * g.mean()
s = 1 / (1 + np.exp(-(Z[te] @ w + b))); pred = s > .5; t = y[te] == 1
prec = (pred & t).sum() / max(1, pred.sum()); rec = (pred & t).sum() / max(1, t.sum())
m = dict(auc=round(float(auc(y[te], s)), 4), precision=round(float(prec), 4), recall=round(float(rec), 4))
json.dump(dict(features=FEATURES, mean=mean.tolist(), std=std.tolist(), w=w.tolist(), b=float(b), metrics=m),
          open("api/model.json", "w"), indent=1)
print("Saved api/model.json", m)
