#!/usr/bin/env python
"""
构建 clean 数据集：
- 读取原始对话 CSV（字段：specific_dialogue_content、interaction_strategy、call_type、is_fraud、fraud_type）。
- 抽取 left 端句子并拼接成 x_clean，按对话输出一行。

运行示例：
    python bin/build_clean.py \
        --data data/训练集结果.csv \
        --output outputs/clean_all.csv
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import List, Optional

import pandas as pd


def log(msg: str) -> None:
    print(f"[info] {msg}")


def normalize_line_breaks(text: str) -> str:
    return text.replace("\r\n", "\n").replace("\r", "\n")


def extract_left_segments(dialogue_text: str) -> List[str]:
    """抽取 left 端轮次文本并切分为句子列表。"""
    if not isinstance(dialogue_text, str):
        return []
    text = normalize_line_breaks(dialogue_text)
    segments: List[str] = []
    current_speaker: Optional[str] = None
    buffer: List[str] = []

    for raw_line in text.split("\n"):
        line = raw_line.strip()
        if not line:
            continue
        speaker_match = re.match(r"^(left|right)\s*[:：]\s*(.*)$", line, flags=re.IGNORECASE)
        if speaker_match:
            speaker = speaker_match.group(1).lower()
            content = speaker_match.group(2).strip()
            if current_speaker == "left" and buffer:
                segments.append(" ".join(buffer))
            current_speaker = speaker
            buffer = [content] if content else []
            continue
        if current_speaker == "left":
            buffer.append(line)

    if current_speaker == "left" and buffer:
        segments.append(" ".join(buffer))

    sentences: List[str] = []
    for seg in segments:
        for sent in re.split(r"[。！？!?；;]+", seg):
            clean = sent.strip(" ，,、.·：:;；!?！？")
            if clean:
                sentences.append(clean)
    return sentences


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="构建 clean 数据集（按对话拼接 left 端）")
    parser.add_argument("--data", type=Path, default=Path("data/训练集结果.csv"), help="原始数据 CSV 路径")
    parser.add_argument("--output", type=Path, default=Path("outputs/clean_all.csv"), help="输出 CSV 路径")
    parser.add_argument("--sample", type=int, default=None, help="仅处理前 N 条对话（调试用）")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    log(f"读取原始数据：{args.data}")
    df = pd.read_csv(args.data, encoding="utf-8-sig")
    if args.sample:
        df = df.head(args.sample)
        log(f"抽样前 {args.sample} 条对话。")

    rows = []
    for idx, row in df.iterrows():
        sentences = extract_left_segments(row.get("specific_dialogue_content", ""))
        x_clean = "。".join(sentences)
        rows.append(
            {
                "id": int(idx),
                "x_clean": x_clean,
                "is_fraud": int(str(row.get("is_fraud")).lower() in {"true", "1", "yes", "y", "是"}),
                "interaction_strategy": row.get("interaction_strategy"),
                "fraud_type": row.get("fraud_type"),
                "call_type": row.get("call_type"),
            }
        )
        if idx < 2:
            log(f"[样例] 对话 {idx} left 句子数：{len(sentences)}，拼接长度：{len(x_clean)}")

    out_df = pd.DataFrame(rows)
    out_df.to_csv(args.output, index=False, encoding="utf-8-sig")
    log(f"已保存 clean 数据：{args.output}")


if __name__ == "__main__":
    main()
