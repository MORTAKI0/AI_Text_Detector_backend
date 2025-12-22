import os, json
from collections import Counter

PATHS = [
    os.getenv("TRAIN_JSONL", ""),
    os.getenv("DEV_JSONL", ""),
]

PATHS = [p for p in PATHS if p.strip()]

if not PATHS:
    raise SystemExit(
        "No JSONL paths provided.\n"
        "Set TRAIN_JSONL and/or DEV_JSONL env vars, e.g.\n"
        "  $env:TRAIN_JSONL='C:\\path\\train.jsonl'\n"
        "  $env:DEV_JSONL='C:\\path\\dev.jsonl'\n"
    )

def to_int_label(v):
    if isinstance(v, bool): return int(v)
    if isinstance(v, int): return v
    if isinstance(v, str):
        s = v.strip().lower()
        if s in ("0","human"): return 0
        if s in ("1","machine","ai","generated"): return 1
    return None

for path in PATHS:
    c = Counter()
    n = 0
    bad = 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            lab = obj.get("label")
            t = to_int_label(lab)
            if t is None:
                bad += 1
                continue
            c[t] += 1
            n += 1
    print(path)
    print(" counts:", dict(c), "total:", n, "bad_labels:", bad)
    print()
