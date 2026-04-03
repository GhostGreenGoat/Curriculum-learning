"""Download OpenThoughts-114k-math and convert to verl training format.
Extracts clean boxed answers from solutions for efficient reward evaluation."""
import os
import re
import json
import pandas as pd
from datasets import load_dataset

OUTPUT_DIR = "/export/home/zhaolei/laiminzhi/data/train/openthoughts_math"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def extract_last_boxed(text):
    """Extract the content from the last \\boxed{...} in the text."""
    idx = text.rfind("\\boxed")
    if idx < 0:
        return None
    i = text.find("{", idx)
    if i < 0:
        return None
    depth = 1
    j = i + 1
    while j < len(text) and depth > 0:
        if text[j] == "{":
            depth += 1
        elif text[j] == "}":
            depth -= 1
        j += 1
    if depth == 0:
        return text[i + 1 : j - 1]
    return None

print("Loading open-r1/OpenThoughts-114k-math...")
ds = load_dataset("open-r1/OpenThoughts-114k-math", split="train")
print(f"Total rows: {len(ds)}")

rows = []
stats = {"total": 0, "has_boxed": 0, "no_boxed_has_solution": 0, "skipped": 0}
source_counts = {}

for i, row in enumerate(ds):
    problem = row.get("problem", "").strip()
    solution = row.get("solution", "").strip()
    source = row.get("source", "openthoughts_math")
    
    if not problem:
        stats["skipped"] += 1
        continue
    
    stats["total"] += 1
    source_counts[source] = source_counts.get(source, 0) + 1

    boxed_answer = extract_last_boxed(solution) if solution else None
    
    if boxed_answer is not None:
        gt = boxed_answer
        stats["has_boxed"] += 1
    elif solution:
        gt = solution
        stats["no_boxed_has_solution"] += 1
    else:
        stats["skipped"] += 1
        continue

    rows.append({
        "prompt": [{"content": problem, "role": "user"}],
        "reward_model": {"ground_truth": gt, "style": "rule"},
        "data_source": source,
        "id": i,
        "ability": "math",
    })

print(f"\nStats:")
print(f"  Total processed: {stats['total']}")
print(f"  Has \\boxed answer: {stats['has_boxed']}")
print(f"  No \\boxed (full solution as GT): {stats['no_boxed_has_solution']}")
print(f"  Skipped: {stats['skipped']}")
print(f"  Converted: {len(rows)}")

print(f"\nSource distribution:")
for s, c in sorted(source_counts.items(), key=lambda x: -x[1]):
    print(f"  {s}: {c}")

df = pd.DataFrame(rows)
out_path = os.path.join(OUTPUT_DIR, "train.parquet")
df.to_parquet(out_path, index=False)
print(f"\nSaved to {out_path}")
print(f"File size: {os.path.getsize(out_path) / 1024 / 1024:.1f} MB")

verify = pd.read_parquet(out_path)
print(f"Verification - shape: {verify.shape}")
for idx in [0, 100, 1000, 5000]:
    gt = verify.iloc[idx]['reward_model']['ground_truth']
    src = verify.iloc[idx]['data_source']
    print(f"  Row {idx} ({src}): GT = {gt[:100]}{'...' if len(gt)>100 else ''}")
