from dotenv import load_dotenv
load_dotenv()  # loads .env from current directory

import os, torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification

m = os.getenv("MODEL_DIR", "./model")

try:
    tok = AutoTokenizer.from_pretrained(m, local_files_only=True, fix_mistral_regex=True)
except TypeError:
    tok = AutoTokenizer.from_pretrained(m, local_files_only=True)

model = AutoModelForSequenceClassification.from_pretrained(m, local_files_only=True)

tests = [
    ("AI-ish", "In conclusion, it is evident that leveraging synergistic strategies can optimize outcomes across multiple domains..."),
    ("Human-ish", "Bro I tried to fix it for like 2 hours and I still don’t get why it crashes 😭 I’ll try again tomorrow."),
]

for name, txt in tests:
    inp = tok(txt, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        logits = model(**inp).logits
        probs = F.softmax(logits, dim=-1).squeeze(0).tolist()
        pred = int(torch.argmax(logits, dim=-1).item())
    print(f"{name}: pred_id={pred} probs={probs}")
