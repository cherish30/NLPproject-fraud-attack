"""
评估指标：Clean Acc / Adv Acc / Drop / ASR / Fidelity。
"""
from __future__ import annotations

import pandas as pd


def accuracy(pred, label) -> float:
    pred_s = pd.Series(pred)
    label_s = pd.Series(label)
    return float((pred_s == label_s).mean())


def compute_metrics(clean_df: pd.DataFrame, adv_df: pd.DataFrame) -> dict:
    # 对齐 id，避免 clean/adv 不同子集导致报错
    common_ids = set(clean_df["id"]).intersection(set(adv_df["id"]))
    clean_df = clean_df[clean_df["id"].isin(common_ids)].reset_index(drop=True)
    adv_df = adv_df[adv_df["id"].isin(common_ids)].reset_index(drop=True)

    clean_acc = accuracy(clean_df["pred"], clean_df["y"])
    adv_acc = accuracy(adv_df["pred"], adv_df["y"])
    drop = clean_acc - adv_acc

    correct_clean = clean_df[clean_df["pred"] == clean_df["y"]]
    merged = correct_clean.merge(adv_df, on="id", suffixes=("_clean", "_adv"))
    asr = float((merged["pred_adv"] != merged["y_adv"]).mean()) if len(merged) else 0.0

    fidelity_pass_rate = float(adv_df.get("fidelity_pass", pd.Series([0]*len(adv_df))).mean())
    fidelity_sim_mean = float(adv_df.get("fidelity_sim", pd.Series([0.0]*len(adv_df))).mean())

    return {
        "clean_acc": round(clean_acc, 4),
        "adv_acc": round(adv_acc, 4),
        "drop": round(drop, 4),
        "asr": round(asr, 4),
        "fidelity_pass_rate": round(fidelity_pass_rate, 4),
        "fidelity_sim_mean": round(fidelity_sim_mean, 4),
    }
