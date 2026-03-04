#!/usr/bin/env python3
"""将 DeepScaleR JSON 数据集转换为 verl GRPO 所需的 parquet 格式。"""

import argparse
import json
import os

import pandas as pd


def convert_deepscaler_to_verl_format(
    input_path: str,
    output_path: str,
    instruction: str = "Let's think step by step and output the final answer within \\boxed{}.",
    data_source: str = "math_dapo",
) -> int:
    """转换 DeepScaleR 格式到 verl 格式。

    Args:
        input_path: 输入的 JSON 文件路径（每行一个 JSON 对象）
        output_path: 输出的 parquet 文件路径
        instruction: 附加到题目后的指令
        data_source: 用于 reward 计算的 data_source（math_dapo 支持 AIME/AMC 格式）

    Returns:
        转换的样本数量
    """
    rows = []
    with open(input_path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                example = json.loads(line)
            except json.JSONDecodeError:
                # 可能是整个文件是一个 JSON 数组
                continue
            problem = example.get("problem", "")
            answer = example.get("answer", "")
            if not problem or not answer:
                continue
            question = f"{problem} {instruction}".strip()
            rows.append({
                "data_source": data_source,
                "prompt": [{"role": "user", "content": question}],
                "ability": "math",
                "reward_model": {
                    "style": "rule",
                    "ground_truth": answer if isinstance(answer, str) else str(answer),
                },
                "extra_info": {"split": "train", "index": idx},
            })

    # 处理整个文件是 JSON 数组的情况
    if not rows:
        with open(input_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            for idx, example in enumerate(data):
                problem = example.get("problem", "")
                answer = example.get("answer", "")
                if not problem or not answer:
                    continue
                question = f"{problem} {instruction}".strip()
                rows.append({
                    "data_source": data_source,
                    "prompt": [{"role": "user", "content": question}],
                    "ability": "math",
                    "reward_model": {
                        "style": "rule",
                        "ground_truth": answer if isinstance(answer, str) else str(answer),
                    },
                    "extra_info": {"split": "train", "index": idx},
                })

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    df.to_parquet(output_path, index=False)
    return len(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        default="/home/ubuntu/date/data/deepscaler/raw/deepscaler.json",
        help="DeepScaleR JSON 输入路径",
    )
    parser.add_argument(
        "--output",
        default="/home/ubuntu/date/data/deepscaler/train.parquet",
        help="输出 parquet 路径",
    )
    args = parser.parse_args()

    # 检查输入格式
    with open(args.input, "r", encoding="utf-8") as f:
        first_char = f.read(1)
    f = open(args.input, "r", encoding="utf-8")
    content = f.read()
    f.close()

    rows = []
    if content.strip().startswith("["):
        # JSON 数组格式
        data = json.loads(content)
        for idx, example in enumerate(data):
            problem = example.get("problem", "")
            answer = example.get("answer", "")
            if not problem or not answer:
                continue
            instruction = "Let's think step by step and output the final answer within \\boxed{}."
            question = f"{problem} {instruction}".strip()
            rows.append({
                "data_source": "math_dapo",
                "prompt": [{"role": "user", "content": question}],
                "ability": "math",
                "reward_model": {
                    "style": "rule",
                    "ground_truth": answer if isinstance(answer, str) else str(answer),
                },
                "extra_info": {"split": "train", "index": idx},
            })
    else:
        # JSONL 格式（每行一个 JSON）
        for idx, line in enumerate(content.strip().split("\n")):
            if not line:
                continue
            try:
                example = json.loads(line)
            except json.JSONDecodeError:
                continue
            problem = example.get("problem", "")
            answer = example.get("answer", "")
            if not problem or not answer:
                continue
            instruction = "Let's think step by step and output the final answer within \\boxed{}."
            question = f"{problem} {instruction}".strip()
            rows.append({
                "data_source": "math_dapo",
                "prompt": [{"role": "user", "content": question}],
                "ability": "math",
                "reward_model": {
                    "style": "rule",
                    "ground_truth": answer if isinstance(answer, str) else str(answer),
                },
                "extra_info": {"split": "train", "index": idx},
            })

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    df.to_parquet(args.output, index=False)
    print(f"✓ 已转换 {len(rows)} 条样本到 {args.output}")


if __name__ == "__main__":
    main()
