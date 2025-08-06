#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
read converted_mllm_data_caption_ko_clean_all.json  →  随机抽样
  · 30000 条写入 ko_train.json
  · 6000  条（与训练集不重合）写入 ko_eval.json
"""

import json
import random
from pathlib import Path

SRC = "converted_mllm_data_caption_ru_clean_all.json"
TRAIN_OUT = "ru_train.json"
EVAL_OUT = "ru_eval.json"
N_TRAIN = 20_000
N_EVAL  = 6_000
SEED = 42               # 为了可复现

def main():
    # 1. 读取源文件（假设整个文件是一个 JSON 列表）
    with open(SRC, "r", encoding="utf-8") as f:
        data = json.load(f)          # list[dict]

    total = len(data)
    assert total >= N_TRAIN + N_EVAL, \
        f"数据不足：需要 {N_TRAIN+N_EVAL} 条，实际只有 {total} 条"

    # 2. 随机抽样（不重合）
    random.seed(SEED)
    indices = list(range(total))
    random.shuffle(indices)

    train_idx = set(indices[:N_TRAIN])
    eval_idx  = set(indices[N_TRAIN:N_TRAIN+N_EVAL])

    train_data = [data[i] for i in train_idx]
    eval_data  = [data[i] for i in eval_idx]

    # 3. 保存到文件
    Path(TRAIN_OUT).write_text(json.dumps(train_data, ensure_ascii=False, indent=2), encoding="utf-8")
    Path(EVAL_OUT).write_text (json.dumps(eval_data , ensure_ascii=False, indent=2), encoding="utf-8")

    print(f" 已生成 {TRAIN_OUT}（{len(train_data)} 条）和 {EVAL_OUT}（{len(eval_data)} 条）")

if __name__ == "__main__":
    main()
