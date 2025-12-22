from pathlib import Path

# where to search (fast + likely places)
roots = [
    Path.cwd(),
    Path.home() / "Downloads",
    Path.home() / "Desktop",
    Path.home() / "OneDrive",
]

targets = {
    "TRAIN": "subtaskA_train_monolingual.jsonl",
    "DEV": "subtaskA_dev_monolingual.jsonl",
}

found = {}

for root in roots:
    if not root.exists():
        continue
    for key, name in targets.items():
        if key in found:
            continue
        for p in root.rglob(name):
            found[key] = p
            break

print("Found:")
for k in ("TRAIN", "DEV"):
    print(f"  {k} =", found.get(k))

if "TRAIN" not in found or "DEV" not in found:
    raise SystemExit(
        "\nCould not find both JSONL files.\n"
        "Fix: search manually in File Explorer for:\n"
        "  subtaskA_train_monolingual.jsonl\n"
        "  subtaskA_dev_monolingual.jsonl\n"
        "Then set env vars with the real full paths.\n"
    )
