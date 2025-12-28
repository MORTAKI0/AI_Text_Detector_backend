from datasets import load_dataset
import pandas as pd

ds = load_dataset("d0rj/SemEval2024-task8", "subtaskA_monolingual", split="dev")
df = ds.to_pandas()[["label","model","source","domain"]]
df["has_model"] = df["model"].fillna("").str.strip().ne("")

print("has_model rate by label (if AI texts have a generator model name, it will be high):")
print(df.groupby("label")["has_model"].mean())

print("\nTop model names for each label (first 10):")
for lab in [0,1]:
    top = df[df["label"]==lab]["model"].fillna("").value_counts().head(10)
    print("\nlabel", lab)
    print(top)




