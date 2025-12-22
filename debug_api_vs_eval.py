from datasets import load_dataset
import numpy as np

from app.inference import analyze_text

# Try to import the "direct full text" proba helper if it exists
try:
    from app.inference import _predict_proba
except Exception:
    _predict_proba = None

def main():
    ds = load_dataset("d0rj/SemEval2024-task8", "subtaskA_monolingual", split="dev")
    # 10 human + 10 AI from dev
    humans = [ex for ex in ds if ex["label"] == 0][:10]
    ais    = [ex for ex in ds if ex["label"] == 1][:10]
    sample = humans + ais

    print("Loaded sample:", len(sample), "(human=0, ai=1)")
    print("-" * 90)

    for ex in sample:
        text = ex["text"]
        true = ex["label"]

        # API-style result (your backend logic)
        r = analyze_text(text)
        api_prob = r.get("prob_ai", None)
        api_label = r.get("label", None)

        # Direct full-text probability (should match eval script style) if available
        direct_prob = _predict_proba(text) if _predict_proba else None

        preview = text.replace("\n", " ")[:90]
        print(f"true={true} | api_label={api_label} api_prob={api_prob:.4f} | direct_prob={direct_prob if direct_prob is not None else 'N/A'}")
        print(f"  text: {preview}")
        print("-" * 90)

    if _predict_proba is None:
        print("\nNOTE: _predict_proba() not found in app.inference.")
        print("If Codex added it under a different name, tell me and I’ll adapt this script.")

if __name__ == "__main__":
    main()
