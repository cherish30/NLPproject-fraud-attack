#!/usr/bin/env python
"""
实验3：分桶分析（按 interaction_strategy 或 rewrite_type 统计 ASR）
输入：clean 预测 CSV、adv 预测 CSV（需含 id, y, pred, rewrite_type, interaction_strategy）
输出：每个分桶的 ASR 表
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def log(msg: str) -> None:
    print(f"[info] {msg}")


def bucket_asr(clean_df: pd.DataFrame, adv_df: pd.DataFrame, bucket_col: str) -> pd.DataFrame:
    # 识别标签列并对齐 id
    clean_label = "y" if "y" in clean_df.columns else ("is_fraud" if "is_fraud" in clean_df.columns else None)
    adv_label = "y" if "y" in adv_df.columns else ("is_fraud" if "is_fraud" in adv_df.columns else None)
    if clean_label is None or adv_label is None:
        raise KeyError("输入缺少标签列（期望 y 或 is_fraud）")
    common_ids = set(clean_df["id"]).intersection(set(adv_df["id"]))
    clean_df = clean_df[clean_df["id"].isin(common_ids)].reset_index(drop=True)
    adv_df = adv_df[adv_df["id"].isin(common_ids)].reset_index(drop=True)

    res = []
    clean_correct = clean_df[clean_df["pred"] == clean_df[clean_label]]
    merged = clean_correct[["id", bucket_col]].merge(
        adv_df[["id", "pred", adv_label, bucket_col]],
        on=["id", bucket_col],
        how="left",
    )
    for b, sub in merged.groupby(bucket_col):
        if len(sub) == 0:
            continue
        asr = (sub["pred"] != sub[adv_label]).mean()
        res.append({"bucket": b, "count": len(sub), "asr": round(float(asr), 4)})
    return pd.DataFrame(res).sort_values(by="asr", ascending=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="实验3：分桶 ASR 分析")
    parser.add_argument("--clean", type=Path, required=True, help="clean 预测 CSV（需含 id, y, pred, interaction_strategy）")
    parser.add_argument("--adv", type=Path, required=True, help="adv 预测 CSV（需含 id, y, pred, rewrite_type, interaction_strategy）")
    parser.add_argument("--bucket", type=str, default="interaction_strategy", help="分桶列名（如 interaction_strategy 或 rewrite_type）")
    parser.add_argument("--output", type=Path, required=True, help="输出 CSV")
    args = parser.parse_args()

    clean_df = pd.read_csv(args.clean, encoding="utf-8-sig")
    adv_df = pd.read_csv(args.adv, encoding="utf-8-sig")
    if args.bucket not in clean_df.columns or args.bucket not in adv_df.columns:
        raise ValueError(f"分桶列 {args.bucket} 在输入中不存在")
    out_df = bucket_asr(clean_df, adv_df, args.bucket)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.output, index=False, encoding="utf-8-sig")
    log(f"分桶 ASR 结果保存：{args.output}")


if __name__ == "__main__":
    main()
