#!/usr/bin/env python
"""
统一评测入口：
- 输入 clean CSV 和 adv CSV（需含 id, y, pred 列；adv 需有 fidelity_* 可选）
- 计算 Clean/Adv Acc, Drop, ASR, Fidelity
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from metrics import compute_metrics


def log(msg: str) -> None:
    print(f"[info] {msg}")


def main() -> None:
    parser = argparse.ArgumentParser(description="评测 Clean/Adv 指标")
    parser.add_argument("--clean", type=Path, required=True, help="clean 预测 CSV（需含 id,y,pred）")
    parser.add_argument("--adv", type=Path, required=True, help="adv 预测 CSV（需含 id,y,pred，可含 fidelity_*）")
    parser.add_argument("--output", type=Path, required=True, help="输出结果 CSV")
    args = parser.parse_args()

    clean_df = pd.read_csv(args.clean, encoding="utf-8-sig")
    adv_df = pd.read_csv(args.adv, encoding="utf-8-sig")
    log(f"clean {len(clean_df)}, adv {len(adv_df)}")

    res = compute_metrics(clean_df, adv_df)
    out_df = pd.DataFrame([res])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.output, index=False, encoding="utf-8-sig")
    log(f"结果：{res}")
    log(f"已保存 {args.output}")


if __name__ == "__main__":
    main()
