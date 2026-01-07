#!/usr/bin/env python
"""
传统分类器（BERT/RoBERTa）二分类：
- 训练：python bin/victim_bert.py --train outputs/clean_train.csv --val outputs/clean_val.csv --save outputs/bert_model
- 预测：python bin/victim_bert.py --predict outputs/clean_test.csv --model outputs/bert_model --output outputs/bert_predictions.csv

输入 CSV 需包含字段：x_clean, is_fraud
"""
from __future__ import annotations

import os

# 提前设环境变量，避免 RTX 4000 P2P 报错/并行警告
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("NCCL_P2P_DISABLE", "1")
os.environ.setdefault("NCCL_IB_DISABLE", "1")

import argparse
from pathlib import Path
from typing import List

import pandas as pd
import torch
from datasets import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)


def log(msg: str) -> None:
    print(f"[info] {msg}")


def tokenize(batch, tokenizer):
    return tokenizer(batch["x_clean"], truncation=True, max_length=512)


def train_model(train_path: Path, val_path: Path, save_dir: Path, model_name: str = "hfl/chinese-roberta-wwm-ext") -> None:
    save_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

    train_ds = Dataset.from_pandas(pd.read_csv(train_path, encoding="utf-8-sig"))
    val_ds = Dataset.from_pandas(pd.read_csv(val_path, encoding="utf-8-sig"))

    # 确保标签列命名为 labels 供 Trainer 计算 loss
    if "is_fraud" in train_ds.column_names:
        train_ds = train_ds.rename_column("is_fraud", "labels")
    if "is_fraud" in val_ds.column_names:
        val_ds = val_ds.rename_column("is_fraud", "labels")

    train_ds = train_ds.map(lambda x: tokenize(x, tokenizer), batched=True)
    val_ds = val_ds.map(lambda x: tokenize(x, tokenizer), batched=True)

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # 兼容不同版本 transformers：旧版不支持 evaluation_strategy/load_best_model_at_end
    # 兼容旧版 transformers：使用最小参数集，评估通过 trainer.evaluate()
    args = TrainingArguments(
        output_dir=str(save_dir / "tmp"),
        num_train_epochs=2,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        learning_rate=2e-5,
        weight_decay=0.01,
        logging_steps=50,
        report_to=[],  # 禁用默认日志后端
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )
    log("开始训练 BERT...")
    trainer.train()
    try:
        trainer.evaluate()
    except Exception:
        pass
    log("保存模型...")
    trainer.save_model(save_dir)
    tokenizer.save_pretrained(save_dir)
    log(f"模型已保存到 {save_dir}")


class VictimBERT:
    def __init__(self, model_dir: Path):
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_dir)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def predict(self, texts: List[str], batch_size: int = 16) -> List[int]:
        preds: List[int] = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            inputs = self.tokenizer(batch, padding=True, truncation=True, max_length=512, return_tensors="pt").to(self.device)
            with torch.no_grad():
                logits = self.model(**inputs).logits
                pred = logits.argmax(dim=-1).cpu().tolist()
                preds.extend(pred)
        return preds


def main() -> None:
    parser = argparse.ArgumentParser(description="BERT/RoBERTa 二分类")
    parser.add_argument("--train", type=Path, help="训练集 CSV 路径")
    parser.add_argument("--val", type=Path, help="验证集 CSV 路径")
    parser.add_argument("--predict", type=Path, help="预测集 CSV 路径")
    parser.add_argument("--output", type=Path, help="预测输出 CSV 路径")
    parser.add_argument("--save", type=Path, help="保存模型目录")
    parser.add_argument("--model", type=Path, help="已训练模型目录（用于预测）")
    parser.add_argument("--model-name", type=str, default="hfl/chinese-roberta-wwm-ext", help="预训练模型名称")
    args = parser.parse_args()

    if args.train and args.val and args.save:
        train_model(args.train, args.val, args.save, model_name=args.model_name)
        return

    if args.predict and args.model and args.output:
        df = pd.read_csv(args.predict, encoding="utf-8-sig")
        clf = VictimBERT(args.model)
        log(f"开始推理 {len(df)} 条样本...")
        preds = clf.predict(df["x_clean"].tolist())
        df_out = df.copy()
        df_out["pred"] = preds
        args.output.parent.mkdir(parents=True, exist_ok=True)
        df_out.to_csv(args.output, index=False, encoding="utf-8-sig")
        log(f"已保存预测：{args.output}")
        return

    parser.error("训练模式需 --train --val --save；预测模式需 --predict --model --output")


if __name__ == "__main__":
    main()
