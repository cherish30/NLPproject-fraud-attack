#!/usr/bin/env python
"""
最小可行对抗改写器：
- 基于 prompt_templates 中的指令调用本地 Qwen3-4B-Instruct 生成候选
- Fidelity 过滤（规则 + 语义）
- 验证 victim 是否被翻转；优先选择成功样本，其次相似度最高

示例：
    python bin/attacker.py \
        --input outputs/clean_test.csv \
        --victim llm \
        --granularity sentence \
        --output outputs/adv_test_sent.csv
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer, pipeline

from fidelity import check_fidelity
from prompt_templates import (
    AG_INSTRUCTIONS,
    AO_TEMPLATE,
    FEW_SHOT_EXAMPLES,
    FEW_SHOT_HINT,
    OI_TEMPLATE,
    SENT_LEVEL_INSTRUCTIONS,
    WORD_LEVEL_INSTRUCTIONS,
)
from victim_llm import VictimLLM
from victim_bert import VictimBERT

MODEL_PATH = "/home/cjn/models/Qwen3-4B-Instruct"


def log(msg: str) -> None:
    print(f"[info] {msg}")


def load_victim(name: str, model_dir: Path | None = None, device: str | int | None = "cuda:0"):
    if name == "llm":
        return VictimLLM(model_path=MODEL_PATH, device=device)
    if name == "bert":
        if model_dir is None:
            raise ValueError("BERT victim 需要 --victim-model-dir")
        return VictimBERT(model_dir)
    raise ValueError(f"未知 victim: {name}")


def build_prompt(text: str, granularity: str, few_shot: bool, target_label: str, extra_inst: str) -> str:
    """
    组合 OI+AO：既强调保真，又提示“让模型更可能判为 {target_label}”
    extra_inst: 具体扰动指令
    """
    base = AO_TEMPLATE + "\n" + OI_TEMPLATE
    base = base.format(text=text)
    guide = f"请在保持语义和关键信息的前提下，改写使模型更可能判断为：{target_label}。"
    prompt = guide + "\n" + extra_inst + "\n" + base
    if few_shot and FEW_SHOT_EXAMPLES:
        shots = "\n".join([f"示例原文：{a}\n示例改写：{b}" for a, b in FEW_SHOT_EXAMPLES])
        prompt = FEW_SHOT_HINT + "\n" + shots + "\n\n" + prompt
    return prompt


def generate_candidates(pipe, text: str, granularity: str, few_shot: bool, ensemble: bool, target_label: str, batch_size: int) -> List[Dict[str, str]]:
    instructs = WORD_LEVEL_INSTRUCTIONS if granularity == "word" else SENT_LEVEL_INSTRUCTIONS
    # ensemble 时加入更多扰动指令
    if ensemble:
        instructs = instructs + AG_INSTRUCTIONS
    else:
        instructs = instructs[:1]

    prompt_info = []
    max_new = min(800, len(text) // 2 + 100)
    for idx, inst in enumerate(instructs):
        prompt = build_prompt(text, granularity, few_shot, target_label, inst)
        for k in range(NUM_CAND):
            prompt_info.append((prompt, f"{granularity}|inst{idx}|k{k}"))

    # 批量生成
    outputs = pipe(
        [p for p, _ in prompt_info],
        max_new_tokens=max_new,
        do_sample=True,
        temperature=1.1,  # 略升随机度
        top_p=0.97,
        top_k=100,
        return_full_text=False,
        batch_size=batch_size,
    )
    candidates: List[Dict[str, str]] = []
    for (prompt, rt), out in zip(prompt_info, outputs):
        # pipeline batch 输出为 List[List[Dict]]；取首个候选
        if isinstance(out, list):
            out = out[0]
        gen = out["generated_text"].strip()
        candidates.append({"prompt": prompt, "text": gen, "rewrite_type": rt})
    return candidates


def attack_row(pipe, victim, row: dict, granularity: str, few_shot: bool, ensemble: bool):
    x_clean = row["x_clean"]
    y = int(row["is_fraud"])
    target_label = "Normal" if y == 1 else "Fraud"
    cands = generate_candidates(pipe, x_clean, granularity, few_shot, ensemble, target_label, BATCH_SIZE)
    best_pass = None
    best_failed = None
    for cand in cands:
        passed, scores = check_fidelity(x_clean, cand["text"])
        if passed:
            pred = victim.predict([cand["text"]])[0]
            attack_success = pred != y
            if attack_success:
                return {
                    "x_adv": cand["text"],
                    "attack_success": True,
                    "fidelity_sim": scores.get("semantic_sim", 0.0),
                    "fidelity_pass": True,
                    "fidelity_sim_raw": scores.get("semantic_sim", 0.0),
                    "fidelity_pass_raw": True,
                    "rewrite_type": cand["rewrite_type"],
                }
            # 记录最佳通过候选
            if best_pass is None or scores.get("semantic_sim", 0.0) > best_pass["fidelity_sim"]:
                best_pass = {
                    "x_adv": cand["text"],
                    "attack_success": False,
                    "fidelity_sim": scores.get("semantic_sim", 0.0),
                    "fidelity_pass": True,
                    "fidelity_sim_raw": scores.get("semantic_sim", 0.0),
                    "fidelity_pass_raw": True,
                    "rewrite_type": cand["rewrite_type"],
                }
        else:
            # 记录最佳未通过候选的相似度，便于回退时保留原始分数
            sim = scores.get("semantic_sim", 0.0)
            if best_failed is None or sim > best_failed["fidelity_sim_raw"]:
                best_failed = {
                    "x_adv": cand["text"],
                    "attack_success": False,
                    "fidelity_sim": sim,  # 原始未通过分
                    "fidelity_pass": False,
                    "fidelity_sim_raw": sim,
                    "fidelity_pass_raw": False,
                    "rewrite_type": cand["rewrite_type"],
                }

    if best_pass:
        return best_pass
    # 回退：使用 clean，标记通过，但保留原始未通过的得分（如有）
    if best_failed:
        return {
            "x_adv": x_clean,
            "attack_success": False,
            "fidelity_sim": 1.0,
            "fidelity_pass": True,
            "fidelity_sim_raw": best_failed.get("fidelity_sim_raw", 0.0),
            "fidelity_pass_raw": False,
            "rewrite_type": "fallback_clean",
        }
    return {
        "x_adv": x_clean,
        "attack_success": False,
        "fidelity_sim": 1.0,
        "fidelity_pass": True,
        "fidelity_sim_raw": 0.0,
        "fidelity_pass_raw": False,
        "rewrite_type": "fallback_clean",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="对抗改写生成")
    parser.add_argument("--input", type=Path, required=True, help="clean 数据 CSV，含 x_clean, is_fraud")
    parser.add_argument("--output", type=Path, required=True, help="输出 adv CSV")
    parser.add_argument("--victim", type=str, default="llm", choices=["llm", "bert"], help="攻击的模型类型")
    parser.add_argument("--victim-model-dir", type=Path, help="BERT 模型目录（victim=bert 时需要）")
    parser.add_argument("--granularity", type=str, default="sentence", choices=["word", "sentence"], help="扰动粒度")
    parser.add_argument("--few-shot", action="store_true", help="是否加入 few-shot 示例")
    parser.add_argument("--ensemble", action="store_true", help="是否多指令 ensemble")
    parser.add_argument("--num-cand", type=int, default=3, help="每条指令生成候选数")
    parser.add_argument("--sample", type=int, default=None, help="仅处理前 N 条样本")
    parser.add_argument("--device", type=str, default="cuda:0", help="生成和受害模型设备，如 cuda:0")
    parser.add_argument("--batch-size", type=int, default=16, help="生成批大小，减轻逐条调用开销")
    args = parser.parse_args()
    global NUM_CAND
    NUM_CAND = max(1, args.num_cand)
    global BATCH_SIZE
    BATCH_SIZE = max(1, args.batch_size)

    df = pd.read_csv(args.input, encoding="utf-8-sig")
    if args.sample:
        df = df.head(args.sample)
    log(f"加载 {len(df)} 条样本，粒度={args.granularity}, few-shot={args.few_shot}, ensemble={args.ensemble}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    # decoder-only，需要左填充 + pad_token
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    gen_pipe = pipeline(
        "text-generation",
        model=MODEL_PATH,
        tokenizer=tokenizer,
        device_map=args.device,
        pad_token_id=tokenizer.pad_token_id,
    )
    victim = load_victim(args.victim, args.victim_model_dir, device=args.device)

    rows = []
    for _, r in tqdm(df.iterrows(), total=len(df), desc="Attacking"):
        adv = attack_row(gen_pipe, victim, r, args.granularity, args.few_shot, args.ensemble)
        rows.append(
            {
                "id": r["id"],
                "interaction_strategy": r.get("interaction_strategy", None),
                "x_clean": r["x_clean"],
                "x_adv": adv["x_adv"],
                "y": int(r["is_fraud"]),
                "attack_success": int(adv["attack_success"]),
                "fidelity_pass": int(adv["fidelity_pass"]),
                "fidelity_sim": adv["fidelity_sim"],
                "fidelity_pass_raw": int(adv.get("fidelity_pass_raw", adv["fidelity_pass"])),
                "fidelity_sim_raw": adv.get("fidelity_sim_raw", adv["fidelity_sim"]),
                "rewrite_type": adv["rewrite_type"],
            }
        )

    out_df = pd.DataFrame(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.output, index=False, encoding="utf-8-sig")
    log(f"已保存对抗样本：{args.output}")


if __name__ == "__main__":
    main()
