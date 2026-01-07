#!/usr/bin/env python
"""
实验2：扰动级别/ few-shot 消融（ensemble 固定开启）
- 四组：word/sentence × few-shot on/off
- 可同时汇总多个 victim（llm/bert/bert_base）。优先读取已有 eval；缺失则现场计算并保存。
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple

import pandas as pd

from exp1_robustness import predict_with_victim
from metrics import compute_metrics


def log(msg: str) -> None:
    print(f"[info] {msg}")


def find_adv_file(out_dir: Path, granularity: str, few: bool) -> Path:
    prefixes = [f"adv_{granularity}"]
    if granularity == "sentence":
        prefixes.append("adv_sent")
    for pref in prefixes:
        for cand in [out_dir / f"{pref}_{int(few)}_1.csv", out_dir / f"{pref}_{int(few)}.csv"]:
            if cand.exists():
                return cand
    raise FileNotFoundError(f"未找到对抗文件 adv_{granularity}_{int(few)}*.csv")


def find_eval_file(out_dir: Path, granularity: str, few: bool, victim: str) -> Path | None:
    prefixes = [f"adv_{granularity}"]
    if granularity == "sentence":
        prefixes.append("adv_sent")
    candidates: List[Path] = []
    for pref in prefixes:
        candidates.extend(
            [
                out_dir / f"{pref}_{int(few)}_eval_{victim}.csv",
                out_dir / f"{pref}_{int(few)}_1_eval_{victim}.csv",
                out_dir / f"{pref}_{int(few)}_eval.csv",
                out_dir / f"{pref}_{int(few)}_1_eval.csv",
            ]
        )
    for c in candidates:
        if c.exists():
            return c
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="实验2：消融汇总")
    parser.add_argument("--clean", type=Path, default=Path("outputs/clean_test.csv"), help="clean 数据 CSV（含 x_clean, is_fraud/y）")
    parser.add_argument("--victims", type=str, default="llm,bert", help="逗号分隔：llm, bert, bert_base")
    parser.add_argument("--bert-model-dir", type=Path, default=Path("outputs/bert_model"), help="微调 BERT 模型目录")
    parser.add_argument("--bert-base-model", type=str, default="hfl/chinese-roberta-wwm-ext", help="bert_base 使用的预训练权重")
    parser.add_argument("--out-dir", type=Path, default=Path("outputs"), help="输出目录")
    parser.add_argument("--batch-size", type=int, default=16, help="预测批大小")
    args = parser.parse_args()

    victims = [v.strip() for v in args.victims.split(",") if v.strip()]
    clean_df = pd.read_csv(args.clean, encoding="utf-8-sig")
    combos: List[Tuple[str, bool]] = [
        ("word", False),
        ("word", True),
        ("sentence", False),
        ("sentence", True),
    ]

    results = []
    for victim in victims:
        log(f"处理 victim={victim}")
        for granularity, few in combos:
            adv_path = find_adv_file(args.out_dir, granularity, few)
            eval_path = find_eval_file(args.out_dir, granularity, few, victim)
            if eval_path is not None:
                log(f"使用已有评测结果 {eval_path}")
                eval_df = pd.read_csv(eval_path, encoding="utf-8-sig")
                res = eval_df.iloc[0].to_dict()
            else:
                log(f"未找到评测结果，现场计算 victim={victim}, adv={adv_path}")
                adv_df = pd.read_csv(adv_path, encoding="utf-8-sig")
                bert_dir = args.bert_model_dir if victim == "bert" else None
                victim_name = victim
                if victim == "bert_base":
                    bert_dir = Path(args.bert_base_model)
                    victim_name = "bert"
                clean_out, adv_out = predict_with_victim(victim_name, clean_df, adv_df, bert_dir, args.batch_size)
                res = compute_metrics(clean_out, adv_out)
                res["victim"] = victim
                # 保存单组 eval 结果，便于复用
                eval_path = args.out_dir / f"adv_{granularity}_{int(few)}_eval_{victim}.csv"
                pd.DataFrame([res]).to_csv(eval_path, index=False, encoding="utf-8-sig")
                log(f"已保存评测结果到 {eval_path}")
            res.update({"granularity": granularity, "few_shot": int(few), "ensemble": 1, "victim": victim})
            results.append(res)

    out_df = pd.DataFrame(results)
    out_path = args.out_dir / "results_exp2.csv"
    out_df.to_csv(out_path, index=False, encoding="utf-8-sig")
    log(f"消融结果保存：{out_path}")


if __name__ == "__main__":
    main()
