from dotenv import load_dotenv
load_dotenv()

import os
import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_dataset
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_DIR = os.getenv("MODEL_DIR", "./model")
N = int(os.getenv("EVAL_N", "1000"))
BATCH = int(os.getenv("EVAL_BS", "16"))

ds = load_dataset("d0rj/SemEval2024-task8", "subtaskA_monolingual", split="dev")
if N > 0:
    ds = ds.select(range(min(N, len(ds))))

try:
    tok = AutoTokenizer.from_pretrained(MODEL_DIR, local_files_only=True, fix_mistral_regex=True)
except TypeError:
    tok = AutoTokenizer.from_pretrained(MODEL_DIR, local_files_only=True)

model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR, local_files_only=True)
model.eval()

texts = ds["text"]
y_true = ds["label"]

y_pred = []
p1_list = []

def chunks(n, bs):
    for i in range(0, n, bs):
        yield i, min(i+bs, n)

with torch.no_grad():
    for i0, i1 in chunks(len(ds), BATCH):
        batch_texts = texts[i0:i1]
        inp = tok(batch_texts, return_tensors="pt", truncation=True, max_length=512, padding=True)
        logits = model(**inp).logits
        probs = F.softmax(logits, dim=-1).cpu().numpy()
        preds = np.argmax(probs, axis=-1)

        y_pred.extend(preds.tolist())
        p1_list.extend(probs[:, 1].tolist())

y_true_np = np.array(y_true)
y_pred_np = np.array(y_pred)
p1 = np.array(p1_list)

acc = accuracy_score(y_true_np, y_pred_np)
cm = confusion_matrix(y_true_np, y_pred_np, labels=[0, 1])

print("MODEL_DIR:", MODEL_DIR)
print("samples :", len(y_true_np))
print("acc     :", round(acc, 4))
print("pred dist:", {0: int((y_pred_np==0).sum()), 1: int((y_pred_np==1).sum())})
print("true dist:", {0: int((y_true_np==0).sum()), 1: int((y_true_np==1).sum())})

print("\nconfusion matrix [rows=true, cols=pred] labels=[0(human),1(machine)]:")
print(cm)

print("\nreport:")
print(classification_report(y_true_np, y_pred_np, digits=4))

# Prob diagnostics
print("\nprob_ai stats:")
print("  mean:", float(p1.mean()), "min:", float(p1.min()), "max:", float(p1.max()))
print("  mean p1 when true=0:", float(p1[y_true_np==0].mean()) if (y_true_np==0).any() else None)
print("  mean p1 when true=1:", float(p1[y_true_np==1].mean()) if (y_true_np==1).any() else None)

try:
    auc = roc_auc_score(y_true_np, p1)
    print("  roc_auc:", round(float(auc), 4))
except Exception as e:
    print("  roc_auc: could not compute:", e)
