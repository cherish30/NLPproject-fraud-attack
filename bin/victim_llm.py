#!/usr/bin/env python
"""
LLM 判别接口（本地 Qwen3-4B-Instruct），仅输出 Fraud / Normal。

示例：
    python bin/victim_llm.py --data outputs/clean_test.csv --output outputs/llm_predictions.csv
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, List

import pandas as pd
from transformers import AutoTokenizer, pipeline

MODEL_PATH = "/home/cjn/models/Qwen3-4B-Instruct"
PROMPT_TMPL = "请判断以下文本是否诈骗，只输出 Fraud 或 Normal。文本：{text}"


def log(msg: str) -> None:
    print(f"[info] {msg}")


class VictimLLM:
    def __init__(self, model_path: str = MODEL_PATH, device: str | int | None = "cuda:0", cache_path: Path | None = None):
        self.model_path = model_path
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.pipe = pipeline(
            "text-generation",
            model=model_path,
            tokenizer=self.tokenizer,
            device_map=(f"cuda:{device}" if isinstance(device, int) or str(device).isdigit() else device),
        )
        self.cache_path = cache_path
        self.cache = {}
        if cache_path and cache_path.exists():
            try:
                self.cache = json.loads(cache_path.read_text(encoding="utf-8"))
            except Exception:
                log("缓存读取失败，忽略。")

    def _normalize_label(self, text: str) -> int:
        t = text.lower()
        if "fraud" in t or "诈骗" in t or "欺诈" in t:
            return 1
        if "normal" in t or "正常" in t or "非诈骗" in t:
            return 0
        return 1  # 保守判 Fraud

    def _infer_one(self, text: str) -> int:
        prompt = PROMPT_TMPL.format(text=text)
        if text in self.cache:
            return self.cache[text]
        if hasattr(self.tokenizer, "apply_chat_template"):
            prompt = self.tokenizer.apply_chat_template(
                [
                    {"role": "system", "content": "你是一个二分类助手。"},
                    {"role": "user", "content": prompt},
                ],
                tokenize=False,
                add_generation_prompt=True,
            )
        out = self.pipe(
            prompt,
            max_new_tokens=10,
            do_sample=False,
            temperature=0.0,
            return_full_text=False,
        )
        gen = out[0]["generated_text"]
        label = self._normalize_label(gen)
        if self.cache_path:
            self.cache[text] = label
        return label

    def predict(self, texts: Iterable[str]) -> List[int]:
        preds = [self._infer_one(t) for t in texts]
        if self.cache_path:
            self.cache_path.write_text(json.dumps(self.cache, ensure_ascii=False, indent=2), encoding="utf-8")
        return preds


def main() -> None:
    parser = argparse.ArgumentParser(description="LLM 判别 (Fraud/Normal)")
    parser.add_argument("--data", type=Path, required=True, help="输入 CSV（需含 x_clean）")
    parser.add_argument("--output", type=Path, required=True, help="输出预测 CSV")
    parser.add_argument("--model-path", type=str, default=MODEL_PATH, help="本地模型路径")
    parser.add_argument("--cache-file", type=Path, default=None, help="缓存文件")
    args = parser.parse_args()

    df = pd.read_csv(args.data, encoding="utf-8-sig")
    victim = VictimLLM(model_path=args.model_path, cache_path=args.cache_file)
    log(f"开始推理 {len(df)} 条样本...")
    preds = victim.predict(df["x_clean"].tolist())
    df_out = df.copy()
    df_out["pred"] = preds
    args.output.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(args.output, index=False, encoding="utf-8-sig")
    log(f"已保存：{args.output}")


if __name__ == "__main__":
    main()
