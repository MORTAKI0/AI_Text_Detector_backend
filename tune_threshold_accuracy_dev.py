from dotenv import load_dotenv
load_dotenv()

import os
import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_dataset
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

MODEL_DIR = os.getenv("MODEL_DIR", "./model")
MAX_LEN   = int(os.getenv("MAX_LEN", "512"))
BS        = int(os.getenv("EVAL_BS", "16"))
N         = int(os.getenv("EVAL_N", "0"))          # 0 = full dev
STEP      = float(os.getenv("THR_STEP", "0.01"))   # fine step
START     = float(os.getenv("THR_START", "0.00"))
END       = float(os.getenv("THR_END", "1.00"))

device = "cuda" if torch.cuda.is_available() else "cpu"
print("device   :", device)
print("MODEL_DIR:", MODEL_DIR)
print("BS       :", BS, "MAX_LEN:", MAX_LEN, "EVAL_N:", N, "STEP:", STEP)

from transformers import AutoTokenizer, AutoModelForSequenceClassification

try:
    tok = AutoTokenizer.from_pretrained(MODEL_DIR, local_files_only=True, fix_mistral_regex=True)
except TypeError:
    tok = AutoTokenizer.from_pretrained(MODEL_DIR, local_files_only=True)

model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR, local_files_only=True).to(device)
model.eval()

print("Loading HF dev...")
ds = load_dataset("d0rj/SemEval2024-task8", "subtaskA_monolingual", split="dev")
if N and N > 0:
    ds = ds.select(range(min(N, len(ds))))

texts = ds["text"]
y_true = np.array(ds["label"], dtype=int)

def iter_batches(n, bs):
    for i in range(0, n, bs):
        yield i, min(i + bs, n)

# compute p(ai) for all examples once
p1 = []
with torch.no_grad():
    for start, end in iter_batches(len(texts), BS):
        batch = texts[start:end]
        enc = tok(batch, return_tensors="pt", truncation=True, max_length=MAX_LEN, padding=True)
        enc = {k: v.to(device) for k, v in enc.items()}
        logits = model(**enc).logits
        probs = F.softmax(logits, dim=-1).detach().cpu().numpy()
        p1.extend(probs[:, 1].tolist())  # index 1 = label 1 (AI) in your project convention

p1 = np.array(p1, dtype=float)
print("Computed probabilities:", len(p1))
print("p1 stats: mean=", float(p1.mean()), "min=", float(p1.min()), "max=", float(p1.max()))

# sweep thresholds
thr_values = np.round(np.arange(START, END + 1e-9, STEP), 2)

best = None
rows = []
for thr in thr_values:
    y_pred = (p1 >= thr).astype(int)
    acc = accuracy_score(y_true, y_pred)
    pred1 = float((y_pred == 1).mean()) * 100.0
    rows.append((thr, acc, pred1))
    # tie-breaker: if same acc, prefer threshold with pred1% closer to 50% (more balanced)
    if best is None or acc > best[1] or (acc == best[1] and abs(pred1 - 50) < abs(best[2] - 50)):
        best = (thr, acc, pred1)

# print top 15 by accuracy
rows_sorted = sorted(rows, key=lambda x: x[1], reverse=True)[:15]
print("\nTop thresholds by accuracy:")
print("thr   acc     pred1%")
for thr, acc, pred1 in rows_sorted:
    print(f"{thr:>4.2f}  {acc:>6.4f}  {pred1:>6.1f}")

best_thr, best_acc, best_pred1 = best
print("\nBEST threshold (by accuracy):")
print("thr=", best_thr, "acc=", round(float(best_acc), 4), "pred1%=", round(float(best_pred1), 1))

# detailed report for best threshold
y_pred_best = (p1 >= best_thr).astype(int)
cm = confusion_matrix(y_true, y_pred_best, labels=[0, 1])

print("\nConfusion matrix [rows=true, cols=pred] labels=[0,1]:")
print(cm)

print("\nClassification report:")
print(classification_report(y_true, y_pred_best, digits=4))

print("\nRecommendation:")
print(f"Set AI_THRESHOLD={best_thr:.2f}")
