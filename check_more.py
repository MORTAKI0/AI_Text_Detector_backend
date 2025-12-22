from dotenv import load_dotenv
load_dotenv()

import os, torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification

m = os.getenv("MODEL_DIR", "./model")

try:
    tok = AutoTokenizer.from_pretrained(m, local_files_only=True, fix_mistral_regex=True)
except TypeError:
    tok = AutoTokenizer.from_pretrained(m, local_files_only=True)

model = AutoModelForSequenceClassification.from_pretrained(m, local_files_only=True)
model.eval()

tests = [
    ("very human short", "lol"),
    ("human chatty", "ok bro wlah ma3rftch علاش كيدير crash 😭"),
    ("human messy", "idk man... i tried 3 times n still broken. maybe tomorrow"),
    ("code", "def add(a,b):\n    return a+b\nprint(add(2,3))"),
    ("formal ai", "In conclusion, the evidence suggests that a multi-faceted approach will optimize outcomes."),
    ("marketing ai", "Leveraging best-in-class solutions enables scalable value creation across stakeholders."),
]

for name, txt in tests:
    inp = tok(txt, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        logits = model(**inp).logits
        probs = F.softmax(logits, dim=-1).squeeze(0).tolist()
        pred = int(torch.argmax(logits, dim=-1).item())
    print(f"{name:15s} pred={pred} p0={probs[0]:.4f} p1={probs[1]:.4f}")
