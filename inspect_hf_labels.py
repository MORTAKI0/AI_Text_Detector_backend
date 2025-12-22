from datasets import load_dataset
ds = load_dataset("d0rj/SemEval2024-task8", "subtaskA_monolingual", split="dev")
print(ds.features)

for lab in [0,1]:
    print("\n=== label", lab, "examples ===")
    c = 0
    for ex in ds:
        if ex["label"] == lab:
            print("-", ex["text"].replace("\n"," ")[:180])
            c += 1
            if c >= 5:
                break
