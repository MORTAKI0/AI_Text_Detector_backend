from dotenv import load_dotenv
load_dotenv()

import os, json
from collections import Counter
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

DEV_PATH = os.getenv("DEV_JSONL", "").strip()
if not DEV_PATH:
    raise SystemExit(
        "DEV_JSONL env var is not set.\n"
        "Example:\n"
        "  $env:DEV_JSONL='C:\\path\\subtaskA_dev_monolingual.jsonl'\n"
    )

N = int(os.getenv("EVAL_N", "500"))
MODEL_DIR = os.getenv("MODEL_DIR", "./model")

def to_int_label(v):
    if isinstance(v, bool): return int(v)
    if isinstance(v, int): return v
    if isinstance(v, str):
        s=v.strip().lower()
        if s in ("0","human"): return 0
        if s in ("1","machine","ai","generated"): return 1
    raise ValueError(f"Unknown label format: {v!r}")

try:
    tok = AutoTokenizer.from_pretrained(MODEL_DIR, local_files_only=True, fix_mistral_regex=True)
except TypeError:
    tok = AutoTokenizer.from_pretrained(MODEL_DIR, local_files_only=True)

model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR, local_files_only=True)
model.eval()

y_true = []
y_pred = []
pred_counts = Counter()

with open(DEV_PATH, "r", encoding="utf-8") as f:
    for line in f:
        if not line.strip():
            continue
        obj = json.loads(line)
        text = obj.get("text") or obj.get("sentence") or obj.get("content")
        label = obj.get("label")
        if text is None or label is None:
            continue

        t = to_int_label(label)

        inp = tok(text, return_tensors="pt", truncation=True, max_length=512)
        with torch.no_grad():
            logits = model(**inp).logits
            pred = int(torch.argmax(logits, dim=-1).item())

        y_true.append(t)
        y_pred.append(pred)
        pred_counts[pred] += 1

        if len(y_true) >= N:
            break

acc = sum(1 for t,p in zip(y_true,y_pred) if t==p) / max(1, len(y_true))
print("samples =", len(y_true))
print("acc     =", round(acc, 4))
print("preds   =", dict(pred_counts))
print("true    =", dict(Counter(y_true)))
