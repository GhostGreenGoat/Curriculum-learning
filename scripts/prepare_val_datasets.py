"""
Convert validation datasets (Numina, Geometry3k) to verl-compatible parquet format.

Required columns:
  - prompt: list of dicts [{"role": "user", "content": "..."}]
  - reward_model: dict {"style": "rule", "ground_truth": "<answer>"}
  - data_source: str (used to route to the correct reward function)

Usage:
  python scripts/prepare_val_datasets.py
"""

import re
import pandas as pd
import numpy as np
from pathlib import Path

DATA_ROOT = Path(__file__).resolve().parent.parent / "data"
OUTPUT_DIR = DATA_ROOT / "test" / "benchmarks"


def extract_boxed_answer(text: str) -> str:
    """Extract the last \\boxed{...} answer from a solution string."""
    idx = text.rfind("\\boxed{")
    if idx < 0:
        return text.strip()

    i = idx
    depth = 0
    for j in range(i, len(text)):
        if text[j] == "{":
            depth += 1
        elif text[j] == "}":
            depth -= 1
            if depth == 0:
                inner = text[idx + len("\\boxed{"):j]
                return inner.strip()
    return text.strip()


def convert_numina(input_path: Path, output_path: Path):
    """Convert Numina test set to verl format."""
    df = pd.read_parquet(input_path)
    print(f"[Numina] Input: {len(df)} rows, sources: {df['source'].unique()}")

    records = []
    for _, row in df.iterrows():
        messages = row["messages"]
        user_msg = None
        for msg in messages:
            if msg["role"] == "user":
                user_msg = msg["content"]
                break

        if user_msg is None:
            continue

        ground_truth = extract_boxed_answer(row["solution"])
        data_source = "numina_" + row["source"]

        records.append({
            "prompt": [{"role": "user", "content": user_msg}],
            "reward_model": {"style": "rule", "ground_truth": ground_truth},
            "data_source": data_source,
            "ability": "math",
            "id": f"numina_{row.name}",
        })

    out_df = pd.DataFrame(records)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_parquet(output_path, index=False)
    print(f"[Numina] Output: {len(out_df)} rows -> {output_path}")
    print(f"  data_source distribution: {out_df['data_source'].value_counts().to_dict()}")


def convert_geometry3k(input_path: Path, output_path: Path):
    """Convert Geometry3k test set to verl format (text-only, no images)."""
    df = pd.read_parquet(input_path)
    print(f"[Geometry3k] Input: {len(df)} rows")

    records = []
    for _, row in df.iterrows():
        problem = row["problem"]
        problem_clean = problem.replace("<image>", "").strip()

        records.append({
            "prompt": [{"role": "user", "content": problem_clean}],
            "reward_model": {"style": "rule", "ground_truth": str(row["answer"])},
            "data_source": "geometry3k",
            "ability": "geometry",
            "id": f"geo3k_{row.name}",
        })

    out_df = pd.DataFrame(records)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_parquet(output_path, index=False)
    print(f"[Geometry3k] Output: {len(out_df)} rows -> {output_path}")
    print(f"  NOTE: Images stripped. Text-only model accuracy will be limited.")


def verify_existing(path: Path, name: str):
    """Verify an existing benchmark parquet has the required columns."""
    if not path.exists():
        print(f"[{name}] WARNING: {path} not found!")
        return False
    df = pd.read_parquet(path)
    required = {"prompt", "reward_model", "data_source"}
    missing = required - set(df.columns)
    if missing:
        print(f"[{name}] WARNING: Missing columns: {missing}")
        return False
    print(f"[{name}] OK: {len(df)} rows, data_source={df['data_source'].unique()}")
    return True


if __name__ == "__main__":
    print("=" * 60)
    print("Preparing validation datasets for verl")
    print("=" * 60)

    print("\n--- Verifying existing benchmarks ---")
    verify_existing(OUTPUT_DIR / "math500.parquet", "MATH-500")
    verify_existing(OUTPUT_DIR / "aime.parquet", "AIME")
    verify_existing(OUTPUT_DIR / "amc.parquet", "AMC")

    print("\n--- Converting Numina ---")
    numina_input = DATA_ROOT / "test" / "numina" / "test.parquet"
    numina_output = OUTPUT_DIR / "numina.parquet"
    if numina_input.exists():
        convert_numina(numina_input, numina_output)
    else:
        print(f"[Numina] Input not found: {numina_input}")

    print("\n--- Converting Geometry3k ---")
    geo3k_input = DATA_ROOT / "test" / "geometry3k" / "test.parquet"
    geo3k_output = OUTPUT_DIR / "geometry3k.parquet"
    if geo3k_input.exists():
        convert_geometry3k(geo3k_input, geo3k_output)
    else:
        print(f"[Geometry3k] Input not found: {geo3k_input}")

    print("\n" + "=" * 60)
    print("Done. Output files in:", OUTPUT_DIR)
