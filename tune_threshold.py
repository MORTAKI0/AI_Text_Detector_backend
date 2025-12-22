from dotenv import load_dotenv
load_dotenv()

import os, numpy as np, torch
import torch.nn.functional as F
from datasets import load_dataset
from sklearn.metrics import f1_score, accuracy_score
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_DIR = os.getenv("MODEL_DIR", "./model")
N = int(os.getenv("EVAL_N", "2000"))
BS = int(os.getenv("EVAL_BS", "16"))
device = "cuda" if torch.cuda.is_available() else "cpu"

ds = load_dataset("d0rj/SemEval2024-task8", "subtaskA_monolingual", split="dev")
ds = ds.select(range(min(N, len(ds))))
texts = ds["text"]
y_true = np.array(ds["label"], dtype=int)

try:
    tok = AutoTokenizer.from_pretrained(MODEL_DIR, local_files_only=True, fix_mistral_regex=True)
except TypeError:
    tok = AutoTokenizer.from_pretrained(MODEL_DIR, local_files_only=True)

model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR, local_files_only=True).to(device)
model.eval()

p1 = []
with torch.no_grad():
    for i in range(0, len(texts), BS):
        batch = texts[i:i+BS]
        enc = tok(batch, return_tensors="pt", truncation=True, max_length=512, padding=True)
        enc = {k:v.to(device) for k,v in enc.items()}
        logits = model(**enc).logits
        probs = F.softmax(logits, dim=-1).detach().cpu().numpy()
        p1.extend(probs[:,1].tolist())

p1 = np.array(p1, dtype=float)

print("Threshold  Acc   F1_macro  F1_label1  Pred1%")
best = None
for thr in [x/100 for x in range(5, 96, 5)]:
    y_pred = (p1 >= thr).astype(int)
    acc = accuracy_score(y_true, y_pred)
    f1m = f1_score(y_true, y_pred, average="macro")
    f11 = f1_score(y_true, y_pred, pos_label=1)
    pred1 = float((y_pred==1).mean())*100
    line = (thr, acc, f1m, f11, pred1)
    if best is None or f1m > best[2]:
        best = line
    print(f"{thr:9.2f} {acc:5.3f} {f1m:8.3f} {f11:9.3f} {pred1:6.1f}")

print("\nBEST by macro-F1:", best)
