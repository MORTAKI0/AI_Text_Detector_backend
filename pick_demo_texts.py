import numpy as np
from datasets import load_dataset

# uses your backend inference helper (full-text probability)
from app.inference import _predict_proba

def main():
    ds = load_dataset("d0rj/SemEval2024-task8", "subtaskA_monolingual", split="dev")

    # compute probs
    probs = []
    for ex in ds:
        p = float(_predict_proba(ex["text"]))
        probs.append(p)

    probs = np.array(probs)
    labels = np.array(ds["label"], dtype=int)

    # pick easiest examples for a clean demo
    human_idx = np.where(labels == 0)[0]
    ai_idx    = np.where(labels == 1)[0]

    # humans with lowest prob_ai
    best_h = human_idx[np.argsort(probs[human_idx])[:3]]
    # AI with highest prob_ai
    best_a = ai_idx[np.argsort(-probs[ai_idx])[:3]]

    print("=== DEMO TEXTS (copy/paste into Swagger) ===\n")

    print("HUMAN examples (should be label=0):")
    for i in best_h:
        t = ds[int(i)]["text"].replace("\n"," ")
        print(f"\n- prob_ai={probs[i]:.3f}\n  {t[:400]}")

    print("\nAI examples (should be label=1):")
    for i in best_a:
        t = ds[int(i)]["text"].replace("\n"," ")
        print(f"\n- prob_ai={probs[i]:.3f}\n  {t[:400]}")

if __name__ == "__main__":
    main()
