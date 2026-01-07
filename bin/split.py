#!/usr/bin/env python
"""
固定 seed 划分 train/val/test。

输入：build_clean.py 生成的 clean CSV（包含 id, x_clean, is_fraud, interaction_strategy 等）
输出：train/val/test 三个 CSV。

运行示例：
    python bin/split.py --data outputs/clean_all.csv --out-dir outputs --seed 42 --train-ratio 0.8 --val-ratio 0.1
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


def log(msg: str) -> None:
    print(f"[info] {msg}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="划分 train/val/test")
    parser.add_argument("--data", type=Path, default=Path("outputs/clean_all.csv"), help="clean 数据路径")
    parser.add_argument("--out-dir", type=Path, default=Path("outputs"), help="输出目录")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--train-ratio", type=float, default=0.8, help="训练集比例")
    parser.add_argument("--val-ratio", type=float, default=0.1, help="验证集比例（余下为测试）")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.data, encoding="utf-8-sig")
    log(f"读取 {len(df)} 条样本。")

    train_df, tmp_df = train_test_split(df, test_size=1 - args.train_ratio, random_state=args.seed, stratify=df["is_fraud"])
    val_size = args.val_ratio / (1 - args.train_ratio)
    val_df, test_df = train_test_split(tmp_df, test_size=1 - val_size, random_state=args.seed, stratify=tmp_df["is_fraud"])

    train_path = args.out_dir / "clean_train.csv"
    val_path = args.out_dir / "clean_val.csv"
    test_path = args.out_dir / "clean_test.csv"

    train_df.to_csv(train_path, index=False, encoding="utf-8-sig")
    val_df.to_csv(val_path, index=False, encoding="utf-8-sig")
    test_df.to_csv(test_path, index=False, encoding="utf-8-sig")

    log(f"train: {len(train_df)}, val: {len(val_df)}, test: {len(test_df)}")
    log(f"保存至 {train_path}, {val_path}, {test_path}")


if __name__ == "__main__":
    main()
