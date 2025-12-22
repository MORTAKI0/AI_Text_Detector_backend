import os
import random

from dotenv import load_dotenv
from datasets import load_dataset

from app import inference
from app.settings import get_settings

load_dotenv()

MODEL_DIR = os.getenv("MODEL_DIR", "./model")
SPLIT = os.getenv("EVAL_SPLIT", "dev")
PER_CLASS = int(os.getenv("DEBUG_N_PER_CLASS", "15"))
SEED = int(os.getenv("DEBUG_SEED", "1337"))


def _pick_samples(dataset, model_value, n, rng):
    models = dataset["model"]
    indices = [i for i, m in enumerate(models) if m == model_value]
    if len(indices) <= n:
        return dataset.select(indices)
    return dataset.select(rng.sample(indices, n))


def main():
    rng = random.Random(SEED)
    settings = get_settings()

    ds = load_dataset(
        "d0rj/SemEval2024-task8",
        "subtaskA_monolingual",
        split=SPLIT,
    )

    human_ds = _pick_samples(ds, "human", PER_CLASS, rng)
    ai_ds = _pick_samples(ds, "bloomz", PER_CLASS, rng)
    sample_ds = human_ds.concatenate(ai_ds)

    rows = []
    for row in sample_ds:
        text = row["text"]
        true_label = int(row["label"])
        model_name = row.get("model", "")
        prob_api_style = inference.analyze_text(text)["prob_ai"]
        prob_direct = inference._predict_proba(text)
        label_api = int(prob_api_style >= settings.AI_THRESHOLD)
        label_direct = int(prob_direct >= settings.AI_THRESHOLD)
        rows.append(
            {
                "true_label": true_label,
                "model": model_name,
                "len": len(text),
                "prob_api": prob_api_style,
                "prob_direct": prob_direct,
                "label_api": label_api,
                "label_direct": label_direct,
            }
        )

    header = (
        "true_label model    len   prob_api  prob_direct  "
        "label_api  label_direct"
    )
    print(header)
    print("-" * len(header))
    for row in rows:
        print(
            f"{row['true_label']:9d} {row['model']:<8} {row['len']:5d} "
            f"{row['prob_api']:9.4f} {row['prob_direct']:11.4f} "
            f"{row['label_api']:9d} {row['label_direct']:12d}"
        )


if __name__ == "__main__":
    main()
