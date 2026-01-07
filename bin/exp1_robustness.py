#!/usr/bin/env python
"""
实验1：Clean vs Adv 鲁棒性评估（最小可行版本）
- 读取 clean_test.csv 和 adv_test.csv（需包含 id,x_clean/x_adv,y）
- 对 victim (bert/llm) 分别在 clean/adv 上预测，输出指标
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from metrics import compute_metrics
from victim_bert import VictimBERT
from victim_llm import VictimLLM


def log(msg: str) -> None:
    print(f"[info] {msg}")


def batch_predict(victim, texts, batch_size: int = 16, desc: str = ""):
    preds = []
    for i in tqdm(range(0, len(texts), batch_size), desc=desc, total=(len(texts) + batch_size - 1) // batch_size):
        preds.extend(victim.predict(texts[i : i + batch_size]))
    return preds


def predict_with_victim(victim_name: str, clean_df: pd.DataFrame, adv_df: pd.DataFrame, bert_model_dir: Path | None, batch_size: int):
    label_col = "y" if "y" in clean_df.columns else "is_fraud"
    if victim_name == "llm":
        victim = VictimLLM()
    elif victim_name == "bert":
        if bert_model_dir is None:
            raise ValueError("BERT 需要 --bert-model-dir")
        victim = VictimBERT(bert_model_dir)
    else:
        raise ValueError("unknown victim")

    clean_preds = batch_predict(victim, clean_df["x_clean"].tolist(), batch_size, desc="Predict clean")
    adv_preds = batch_predict(victim, adv_df["x_adv"].tolist(), batch_size, desc="Predict adv")

    clean_out = clean_df[["id", "x_clean", label_col]].copy()
    clean_out = clean_out.rename(columns={label_col: "y"})
    adv_label_col = label_col if label_col in adv_df.columns else ("y" if "y" in adv_df.columns else None)
    if adv_label_col is None:
        raise KeyError("adv 数据缺少标签列（期望 is_fraud 或 y）")
    adv_cols = ["id", "x_adv", adv_label_col]
    for c in ["fidelity_pass", "fidelity_sim", "fidelity_pass_raw", "fidelity_sim_raw", "rewrite_type", "x_clean"]:
        if c in adv_df.columns:
            adv_cols.append(c)
    adv_out = adv_df[adv_cols].copy()
    adv_out = adv_out.rename(columns={adv_label_col: "y"})
    adv_out["pred"] = adv_preds
    clean_out["pred"] = clean_preds
    return clean_out, adv_out


def main() -> None:
    parser = argparse.ArgumentParser(description="实验1：鲁棒性评估")
    parser.add_argument("--clean", type=Path, required=True, help="clean 数据 CSV（id,x_clean,y）")
    parser.add_argument("--adv", type=Path, required=True, help="adv 数据 CSV（id,x_adv,y）")
    parser.add_argument("--victim", type=str, default="llm", choices=["llm", "bert"], help="victim 类型")
    parser.add_argument("--bert-model-dir", type=Path, help="BERT 模型目录")
    parser.add_argument("--output", type=Path, required=True, help="结果 CSV")
    parser.add_argument("--batch-size", type=int, default=16, help="批量预测大小")
    parser.add_argument("--error-cases", type=Path, help="可选，保存攻击成功样例 jsonl")
    args = parser.parse_args()

    clean_df = pd.read_csv(args.clean, encoding="utf-8-sig")
    adv_df = pd.read_csv(args.adv, encoding="utf-8-sig")
    clean_out, adv_out = predict_with_victim(args.victim, clean_df, adv_df, args.bert_model_dir, args.batch_size)

    res = compute_metrics(clean_out, adv_out)
    res["victim"] = args.victim
    out_df = pd.DataFrame([res])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.output, index=False, encoding="utf-8-sig")
    log(f"结果：{res}")
    log(f"保存到 {args.output}")

    if args.error_cases:
        merged = clean_out.merge(adv_out, on="id", suffixes=("_clean", "_adv"))
        errs = merged[(merged["pred_clean"] == merged["y"]) & (merged["pred_adv"] != merged["y"])]
        if not errs.empty:
            cols = [
                "id",
                "y",
                "x_clean_clean",
                "x_adv",
                "pred_clean",
                "pred_adv",
            ]
            for c in ["fidelity_pass", "fidelity_sim", "rewrite_type"]:
                col = f"{c}" if c in errs.columns else None
                if col:
                    cols.append(col)
            subset = errs[cols].rename(columns={"x_clean_clean": "x_clean"})
            args.error_cases.parent.mkdir(parents=True, exist_ok=True)
            subset.to_json(args.error_cases, orient="records", force_ascii=False, lines=True)
            log(f"攻击成功样例保存：{args.error_cases}，数量 {len(subset)}")
        else:
            log("无攻击成功样例可保存。")


if __name__ == "__main__":
    main()
