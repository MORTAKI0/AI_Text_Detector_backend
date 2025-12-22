from dotenv import load_dotenv
load_dotenv()

import os
import random
import numpy as np
from datasets import load_dataset

# uses backend inference helper (full-text proba)
from app.inference import _predict_proba

N_PER_LABEL = int(os.getenv("N_PER_LABEL", "200"))  # 200 human + 200 ai
TOPK = int(os.getenv("TOPK", "3"))
SEED = int(os.getenv("SEED", "42"))
MAX_CHARS = int(os.getenv("MAX_CHARS", "900"))

def clip(t: str) -> str:
    t = (t or "").strip()
    return t[:MAX_CHARS] if MAX_CHARS > 0 else t

def main():
    ds = load_dataset("d0rj/SemEval2024-task8", "subtaskA_monolingual", split="dev")
    labels = ds["label"]

    idx_h = [i for i, y in enumerate(labels) if int(y) == 0]
    idx_a = [i for i, y in enumerate(labels) if int(y) == 1]

    rng = random.Random(SEED)
    rng.shuffle(idx_h)
    rng.shuffle(idx_a)

    idx = idx_h[:N_PER_LABEL] + idx_a[:N_PER_LABEL]

    probs = []
    ys = []
    texts = []

    print(f"Scoring {len(idx)} samples total (human={N_PER_LABEL}, ai={N_PER_LABEL}) ...")
    for i in idx:
        ex = ds[int(i)]
        t = clip(ex["text"])
        p = float(_predict_proba(t))
        probs.append(p)
        ys.append(int(ex["label"]))
        texts.append(t)

    probs = np.array(probs)
    ys = np.array(ys)

    human_mask = (ys == 0)
    ai_mask = (ys == 1)

    # easiest: humans lowest prob, AI highest prob
    best_h = np.argsort(probs[human_mask])[:TOPK]
    best_a = np.argsort(-probs[ai_mask])[:TOPK]

    human_texts = np.array(texts, dtype=object)[human_mask]
    ai_texts    = np.array(texts, dtype=object)[ai_mask]
    human_probs = probs[human_mask]
    ai_probs    = probs[ai_mask]

    print("\n=== DEMO TEXTS (copy/paste into Swagger) ===\n")

    print("HUMAN examples (should be label=0):")
    for j in best_h:
        print(f"\n- prob_ai={human_probs[j]:.3f}\n  {human_texts[j].replace('\\n',' ')[:450]}")

    print("\nAI examples (should be label=1):")
    for j in best_a:
        print(f"\n- prob_ai={ai_probs[j]:.3f}\n  {ai_texts[j].replace('\\n',' ')[:450]}")

if __name__ == "__main__":
    main()
