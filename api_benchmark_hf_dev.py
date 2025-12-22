import os
import time
import random
import requests
import numpy as np
from datasets import load_dataset
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

API = os.getenv("API_URL", "http://127.0.0.1:8000").rstrip("/")
EMAIL = os.getenv("EMAIL", "user22@example.com")
PASSWORD = os.getenv("PASSWORD", "abdo1234")

N_PER_LABEL = int(os.getenv("N_PER_LABEL", "50"))     # 50 + 50 = 100 total
MAX_CHARS   = int(os.getenv("MAX_CHARS", "1600"))     # truncate long texts
TIMEOUT     = int(os.getenv("TIMEOUT", "180"))        # read timeout seconds
SLEEP_MS    = int(os.getenv("SLEEP_MS", "250"))       # pause between requests
RETRIES     = int(os.getenv("RETRIES", "5"))          # retry on network/timeouts
SEED        = int(os.getenv("SEED", "42"))

def clip_text(t: str) -> str:
    t = (t or "").strip()
    if MAX_CHARS > 0 and len(t) > MAX_CHARS:
        return t[:MAX_CHARS]
    return t

def wait_for_api(session: requests.Session, seconds: int = 30) -> None:
    """Wait until the API responds (use /docs or /openapi.json)."""
    deadline = time.time() + seconds
    last_err = None
    while time.time() < deadline:
        try:
            r = session.get(f"{API}/openapi.json", timeout=(5, 10))
            if r.status_code == 200:
                return
        except Exception as e:
            last_err = e
        time.sleep(0.5)
    raise RuntimeError(f"API not reachable at {API} (last error: {last_err})")

def request_with_retries(fn, *, what: str):
    last = None
    for attempt in range(1, RETRIES + 1):
        try:
            return fn()
        except (requests.exceptions.ConnectionError,
                requests.exceptions.ReadTimeout,
                requests.exceptions.Timeout) as e:
            last = e
            backoff = min(2 ** (attempt - 1), 8)
            print(f"[{what}] retry {attempt}/{RETRIES} after {backoff}s بسبب: {type(e).__name__}")
            time.sleep(backoff)
        except requests.HTTPError as e:
            # For login/analyze we usually want to stop on real 4xx errors
            raise
    raise last

def login(session: requests.Session) -> str:
    def _do():
        r = session.post(
            f"{API}/auth/login",
            json={"email": EMAIL, "password": PASSWORD},
            timeout=(10, TIMEOUT),
        )
        r.raise_for_status()
        return r.json()["access_token"]
    return request_with_retries(_do, what="login")

def analyze(session: requests.Session, token: str, text: str) -> dict:
    def _do():
        r = session.post(
            f"{API}/analyze",
            headers={"Authorization": f"Bearer {token}"},
            json={"text": text},
            timeout=(10, TIMEOUT),
        )
        r.raise_for_status()
        return r.json()
    return request_with_retries(_do, what="analyze")

def sample_indices(labels, target_label: int, n: int, seed: int):
    idx = [i for i, y in enumerate(labels) if int(y) == target_label]
    rng = random.Random(seed + target_label)
    rng.shuffle(idx)
    return idx[:min(n, len(idx))]

def main():
    print(f"API={API}")
    print(f"EMAIL={EMAIL}")
    print(f"N_PER_LABEL={N_PER_LABEL} => total={2*N_PER_LABEL}")
    print(f"MAX_CHARS={MAX_CHARS} TIMEOUT={TIMEOUT}s SLEEP_MS={SLEEP_MS} RETRIES={RETRIES} SEED={SEED}")

    session = requests.Session()
    # wait until server is up (prevents WinError 10061)
    wait_for_api(session, seconds=30)

    print("Loading HF dev split...")
    ds = load_dataset("d0rj/SemEval2024-task8", "subtaskA_monolingual", split="dev")
    labels = ds["label"]

    idx_h = sample_indices(labels, 0, N_PER_LABEL, SEED)
    idx_a = sample_indices(labels, 1, N_PER_LABEL, SEED)
    idx = idx_h + idx_a

    print(f"Sample size: human={len(idx_h)} ai={len(idx_a)} total={len(idx)}")
    print("Logging in...")
    token = login(session)

    y_true, y_pred, p_ai = [], [], []
    errors = 0
    mistakes = []

    for k, i in enumerate(idx, start=1):
        ex = ds[int(i)]
        true = int(ex["label"])
        text = clip_text(ex["text"])

        try:
            out = analyze(session, token, text)
        except Exception as e:
            errors += 1
            print(f"[ERROR] item {k}/{len(idx)} failed: {type(e).__name__}: {e}")
            # continue so you still get a report
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
                "preview": text.replace("\\n", " ")[:140],
            })

        if SLEEP_MS > 0:
            time.sleep(SLEEP_MS / 1000)

        if k % 10 == 0 or k == len(idx):
            print(f"progress {k}/{len(idx)} (ok={len(y_true)} errors={errors})")

    if len(y_true) == 0:
        print("No successful samples. The API is timing out/crashing. Increase TIMEOUT/SLEEP_MS or reduce MAX_CHARS.")
        return

    y_true = np.array(y_true, dtype=int)
    y_pred = np.array(y_pred, dtype=int)
    p_ai = np.array(p_ai, dtype=float)

    acc = accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    print("\\n=== RESULTS (label 0=Human, 1=AI) ===")
    print("successful:", len(y_true), "errors:", errors)
    print("accuracy:", round(float(acc), 4))
    print("\\nconfusion matrix [rows=true, cols=pred] labels=[0,1]:")
    print(cm)

    print("\\nclassification report:")
    print(classification_report(y_true, y_pred, digits=4))

    print("\\nprob_ai stats:")
    print("  mean prob_ai:", float(p_ai.mean()))
    print("  mean prob_ai | true=0:", float(p_ai[y_true == 0].mean()) if (y_true == 0).any() else None)
    print("  mean prob_ai | true=1:", float(p_ai[y_true == 1].mean()) if (y_true == 1).any() else None)

    print(f"\\nMisclassifications: {len(mistakes)}")
    for m in sorted(mistakes, key=lambda x: x["prob_ai"], reverse=True)[:10]:
        print(f"- true={m['true']} pred={m['pred']} prob_ai={m['prob_ai']:.3f} text='{m['preview']}'")

if __name__ == "__main__":
    main()
