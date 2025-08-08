#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
read JSON file → random sampling
  · N_TRAIN 条写入训练集文件
  · N_EVAL 条写入验证集文件，且不与训练集重合
"""

import json
import random
from pathlib import Path
import argparse

def main(src, train_out, eval_out, n_train, n_eval, seed):
    # 1. 读取源文件（假设整个文件是一个 JSON 列表）
    with open(src, "r", encoding="utf-8") as f:
        data = json.load(f)          # list[dict]

    total = len(data)
    assert total >= n_train + n_eval, \
        f"数据不足：需要 {n_train + n_eval} 条，实际只有 {total} 条"

    # 2. 随机抽样（不重合）
    random.seed(seed)
    indices = list(range(total))
    random.shuffle(indices)

    train_idx = set(indices[:n_train])
    eval_idx  = set(indices[n_train:n_train + n_eval])

    train_data = [data[i] for i in train_idx]
    eval_data  = [data[i] for i in eval_idx]

    # 3. 保存到文件
    Path(train_out).write_text(json.dumps(train_data, ensure_ascii=False, indent=2), encoding="utf-8")
    Path(eval_out).write_text(json.dumps(eval_data, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"已生成 {train_out}（{len(train_data)} 条）和 {eval_out}（{len(eval_data)} 条）")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="从 JSON 数据中随机抽样生成训练和验证集")
    parser.add_argument("--src", type=str, required=True, help="输入 JSON 文件路径")
    parser.add_argument("--train_out", type=str, required=True, help="输出训练集 JSON 文件路径")
    parser.add_argument("--eval_out", type=str, required=True, help="输出验证集 JSON 文件路径")
    parser.add_argument("--n_train", type=int, default=20000, help="训练集样本数（默认 20000）")
    parser.add_argument("--n_eval", type=int, default=6000, help="验证集样本数（默认 6000）")
    parser.add_argument("--seed", type=int, default=42, help="随机种子（默认 42）")

    args = parser.parse_args()

    main(args.src, args.train_out, args.eval_out, args.n_train, args.n_eval, args.seed)
