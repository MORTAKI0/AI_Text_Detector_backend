import os
import time
import random
import requests
import numpy as np
from datasets import load_dataset
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

API = os.getenv("API_URL", "http://127.0.0.1:8000")
EMAIL = os.getenv("EMAIL", "user22@example.com")
PASSWORD = os.getenv("PASSWORD", "abdo1234")

# 100 total => 50 human + 50 AI by default
N_PER_LABEL = int(os.getenv("N_PER_LABEL", "50"))

# Avoid extremely long samples that make /analyze slow (segment scoring)
MAX_CHARS = int(os.getenv("MAX_CHARS", "1600"))

# Request timeout is seconds (read timeout). Increase for slow samples.
TIMEOUT = float(os.getenv("TIMEOUT", "180"))

# Be gentle to the API
SLEEP_MS = int(os.getenv("SLEEP_MS", "30"))

# Retry transient failures
RETRIES = int(os.getenv("RETRIES", "4"))
BACKOFF_S = float(os.getenv("BACKOFF_S", "0.75"))

SEED = int(os.getenv("SEED", "42"))

def login(session: requests.Session) -> str:
    r = session.post(f"{API}/auth/login", json={"email": EMAIL, "password": PASSWORD}, timeout=TIMEOUT)
    r.raise_for_status()
    return r.json()["access_token"]

def analyze_with_retries(session: requests.Session, token: str, text: str) -> dict | None:
    last_err = None
    for attempt in range(1, RETRIES + 1):
        try:
            r = session.post(
                f"{API}/analyze",
                headers={"Authorization": f"Bearer {token}"},
                json={"text": text},
                timeout=TIMEOUT,
            )
            r.raise_for_status()
            return r.json()
        except (requests.exceptions.ReadTimeout, requests.exceptions.ConnectionError, requests.exceptions.Timeout) as e:
            last_err = e
            # exponential-ish backoff
            time.sleep(BACKOFF_S * attempt)
        except requests.exceptions.HTTPError as e:
            # If the server returns a real HTTP error, don't retry forever.
            # But retry 401 once (token issues) by re-login outside if you want.
            last_err = e
            break

    print(f"[WARN] analyze failed after {RETRIES} retries: {last_err}")
    return None

def pick_indices(ds, label: int, n: int, max_chars: int, seed: int):
    candidates = []
    for i, ex in enumerate(ds):
        if int(ex["label"]) != label:
            continue
        if len(ex["text"]) > max_chars:
            continue
        candidates.append(i)

    rng = random.Random(seed + label)
    rng.shuffle(candidates)
    return candidates[: min(n, len(candidates))]

def main():
    print(f"API={API}")
    print(f"EMAIL={EMAIL}")
    print(f"N_PER_LABEL={N_PER_LABEL}  MAX_CHARS={MAX_CHARS}  TIMEOUT={TIMEOUT}s")
    print(f"SLEEP_MS={SLEEP_MS}  RETRIES={RETRIES}  BACKOFF_S={BACKOFF_S}  SEED={SEED}")

    print("Loading HF dev split...")
    ds = load_dataset("d0rj/SemEval2024-task8", "subtaskA_monolingual", split="dev")

    idx_h = pick_indices(ds, label=0, n=N_PER_LABEL, max_chars=MAX_CHARS, seed=SEED)
    idx_a = pick_indices(ds, label=1, n=N_PER_LABEL, max_chars=MAX_CHARS, seed=SEED)
    idx = idx_h + idx_a

    print(f"Sample size: human={len(idx_h)} ai={len(idx_a)} total={len(idx)}")
    if len(idx) == 0:
        print("No samples selected. Try increasing MAX_CHARS.")
        return

    with requests.Session() as session:
        print("Logging in...")
        token = login(session)

        y_true, y_pred, p_ai = [], [], []
        mistakes = []
        failures = 0

        for k, i in enumerate(idx, start=1):
            ex = ds[int(i)]
            true = int(ex["label"])
            text = ex["text"]

            out = analyze_with_retries(session, token, text)
            if out is None:
                failures += 1
                # keep going instead of crashing
                continue

            pred = int(out["label"])
            prob = float(out["prob_ai"])

            y_true.append(true)
            y_pred.append(pred)
            p_ai.append(prob)

            if pred != true:
                mistakes.append({
                    "true": true,
                    "pred": pred,
                    "prob_ai": prob,
                    "preview": text.replace("\n", " ")[:140],
                })

            if SLEEP_MS > 0:
                time.sleep(SLEEP_MS / 1000.0)

            if k % 10 == 0 or k == len(idx):
                print(f"progress {k}/{len(idx)} (failures so far: {failures})")

        if len(y_true) == 0:
            print("All requests failed. Check the API is running and credentials are correct.")
            return

        y_true = np.array(y_true, dtype=int)
        y_pred = np.array(y_pred, dtype=int)
        p_ai = np.array(p_ai, dtype=float)

        acc = accuracy_score(y_true, y_pred)
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

        print("\n=== RESULTS (label 0=Human, 1=AI) ===")
        print("processed:", len(y_true), "skipped/failed:", failures)
        print("accuracy:", round(float(acc), 4))
        print("\nconfusion matrix [rows=true, cols=pred] labels=[0,1]:")
        print(cm)

        print("\nclassification report:")
        print(classification_report(y_true, y_pred, digits=4))

        print("\nprob_ai stats:")
        print("  mean prob_ai:", float(p_ai.mean()))
        print("  mean prob_ai | true=0:", float(p_ai[y_true == 0].mean()) if (y_true == 0).any() else None)
        print("  mean prob_ai | true=1:", float(p_ai[y_true == 1].mean()) if (y_true == 1).any() else None)

        print(f"\nMisclassifications: {len(mistakes)} / {len(y_true)}")
        for m in sorted(mistakes, key=lambda x: x["prob_ai"], reverse=True)[:10]:
            print(f"- true={m['true']} pred={m['pred']} prob_ai={m['prob_ai']:.3f} text='{m['preview']}'")

if __name__ == "__main__":
    main()
