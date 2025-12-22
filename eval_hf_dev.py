from dotenv import load_dotenv
load_dotenv()

import os
import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_dataset
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_DIR = os.getenv("MODEL_DIR", "./model")
N = int(os.getenv("EVAL_N", "1000"))   # keep small on CPU; increase later
BATCH = int(os.getenv("EVAL_BS", "16"))

# SemEval 2024 Task 8 dataset on HF:
# subset: subtaskA_monolingual ; split: dev
ds = load_dataset("d0rj/SemEval2024-task8", "subtaskA_monolingual", split="dev")

# optional: limit samples for speed
if N > 0:
    ds = ds.select(range(min(N, len(ds))))

try:
    tok = AutoTokenizer.from_pretrained(MODEL_DIR, local_files_only=True, fix_mistral_regex=True)
except TypeError:
    tok = AutoTokenizer.from_pretrained(MODEL_DIR, local_files_only=True)

model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR, local_files_only=True)
model.eval()

y_true = []
y_pred = []
p1_list = []

def chunks(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i+n]

texts = ds["text"]
labels = ds["label"]

with torch.no_grad():
    for batch_idx in chunks(list(range(len(ds))), BATCH):
        batch_texts = [texts[i] for i in batch_idx]
        batch_labels = [labels[i] for i in batch_idx]

        inp = tok(batch_texts, return_tensors="pt", truncation=True, max_length=512, padding=True)
        logits = model(**inp).logits
        probs = F.softmax(logits, dim=-1).cpu().numpy()  # [bs,2]

        preds = np.argmax(probs, axis=-1).tolist()
        y_true.extend(batch_labels)
        y_pred.extend(preds)
        p1_list.extend(probs[:, 1].tolist())

acc = accuracy_score(y_true, y_pred)
cm = confusion_matrix(y_true, y_pred, labels=[0,1])

print("MODEL_DIR:", MODEL_DIR)
print("samples :", len(y_true))
print("acc     :", round(acc, 4))
print("pred dist:", {0: int((np.array(y_pred)==0).sum()), 1: int((np.array(y_pred)==1).sum())})
print("true dist:", {0: int((np.array(y_true)==0).sum()), 1: int((np.array(y_true)==1).sum())})
print("\nconfusion matrix [rows=true, cols=pred] labels=[0(human),1(machine)]:")
print(cm)
print("\nreport:")
print(classification_report(y_true, y_pred, digits=4))

print("\nprob_ai (p(label=1)) quick stats:")
p1 = np.array(p1_list)
print("  mean:", float(p1.mean()), "min:", float(p1.min()), "max:", float(p1.max()))
