# NLPproject-fraud-attack

## 目录结构（核心）
```
bin/                  # 脚本
  build_clean.py      # 构建 clean 数据（拼接 left）
  split.py            # 划分 train/val/test
  attacker.py         # 对抗改写生成（word/sent，AO+ensemble）
  fidelity.py         # 保真规则+相似度
  prompt_templates.py # 攻击模板与 few-shot 示例
  victim_llm.py       # 本地 Qwen3-4B-Instruct 分类 Fraud/Normal
  victim_bert.py      # BERT/RoBERTa 微调与预测
  exp1_robustness.py  # Clean vs Adv 指标 (Acc/Drop/ASR/Fidelity)
  exp2_ablation.py    # 消融汇总（word/sent × few-shot，ensemble 固定开）
  exp3_analysis.py    # 分桶分析（需带 pred 的 clean/adv 预测文件）
  metrics.py, run_eval.py
data/                 # 原始数据：训练集结果.csv 等
outputs/              # 中间与结果文件（clean/adv/eval）
logs/                 # 日志（nohup 输出）
```

## 环境
- Python 3.10+
- 主要依赖：`pandas`, `numpy`, `transformers`, `torch`, `datasets`, `tqdm`, `sentence-transformers`, `scikit-learn`
- 本地 LLM：`/home/cjn/models/Qwen3-4B-Instruct`（transformers pipeline）
- GPU: 单卡可通过 `--device cuda:0` 指定；BERT 训练可用 `CUDA_VISIBLE_DEVICES`.

---

## 1. 数据准备（clean）
```bash
# 生成 clean_all.csv（拼 left 端为 x_clean）
python bin/build_clean.py --data data/训练集结果.csv --output outputs/clean_all.csv

# 划分 train/val/test
python bin/split.py --data outputs/clean_all.csv --out-dir outputs
# 结果：outputs/clean_train.csv, clean_val.csv, clean_test.csv
```

---

## 2. 对抗攻击（生成 adv）
示例：sentence 粒度 + ensemble + num-cand=10，跑完整 test：
```bash
nohup python -u bin/attacker.py \
  --input outputs/clean_test.csv \
  --output outputs/adv_sent_0.csv \
  --victim llm \
  --granularity sentence \
  --device cuda:0 \
  --num-cand 10 \
  --batch-size 32 \
  --ensemble \
  > logs/att_sent_0.log 2>&1 &
```
其他常用组合（few-shot 开/关；word 粒度）：
```bash
# word, few-shot 关
nohup python -u bin/attacker.py --input outputs/clean_test.csv --output outputs/adv_word_0.csv --victim llm --granularity word --device cuda:1 --num-cand 10 --batch-size 32 --ensemble > logs/att_word_0.log 2>&1 &
# word, few-shot 开
nohup python -u bin/attacker.py --input outputs/clean_test.csv --output outputs/adv_word_1.csv --victim llm --granularity word --few-shot --device cuda:2 --num-cand 10 --batch-size 32 --ensemble > logs/att_word_1.log 2>&1 &
# sentence, few-shot 开
nohup python -u bin/attacker.py --input outputs/clean_test.csv --output outputs/adv_sent_1.csv --victim llm --granularity sentence --few-shot --device cuda:3 --num-cand 10 --batch-size 32 --ensemble > logs/att_sent_1.log 2>&1 &
```
输出列：`id, interaction_strategy, x_clean, x_adv, y/is_fraud, attack_success, fidelity_pass(_raw), fidelity_sim(_raw), rewrite_type`。回退会保留原始保真分。

---

## 3. 实验一：鲁棒性评测（Acc/Drop/ASR/Fidelity）
示例（LLM，sentence 粒度 adv）：
```bash
nohup python -u bin/exp1_robustness.py \
  --clean outputs/clean_test.csv \
  --adv outputs/adv_sent_0.csv \
  --victim llm \
  --batch-size 16 \
  --output outputs/adv_sent_0_eval_llm.csv \
  > logs/eval_sent_0_llm.log 2>&1 &
```
示例（已训练 BERT）：
```bash
nohup python -u bin/exp1_robustness.py \
  --clean outputs/clean_test.csv \
  --adv outputs/adv_sent_0.csv \
  --victim bert \
  --bert-model-dir outputs/bert_model \
  --batch-size 16 \
  --output outputs/adv_sent_0_eval_bert.csv \
  > logs/eval_sent_0_bert.log 2>&1 &
```

---

## 4. 实验二：消融（word/sent × few-shot，ensemble=1）
`exp2_ablation.py` 会对四组组合（word/sent × few-shot on/off）进行评测汇总：
- 先查找已有评测结果：`outputs/adv_{granularity}_{few}_eval_{victim}.csv`（兼容 `adv_sent_*` 简写）。
- 如缺少对应 eval，会报错提示，需先按实验一生成 eval 文件。

运行示例（LLM）：
```bash
nohup python -u bin/exp2_ablation.py --victim llm > logs/exp2_llm.log 2>&1 &
# 输出：outputs/results_exp2.csv
```
如需 BERT/BERT_BASE 受害者，先跑对应 eval，再改 `--victim bert` 或 `bert_base`。

---

## 5. 训练/对比 BERT（可选）
训练（示例，2 epoch）：
```bash
CUDA_VISIBLE_DEVICES=3 nohup python -u bin/victim_bert.py \
  --train outputs/clean_train.csv \
  --val outputs/clean_val.csv \
  --save outputs/bert_model \
  > logs/bert_train.log 2>&1 &
```
未微调基线评测（对比用）：
```bash
nohup python -u bin/exp1_robustness.py \
  --clean outputs/clean_test.csv \
  --adv outputs/adv_sent_0.csv \
  --victim bert \
  --bert-model-dir hfl/chinese-roberta-wwm-ext \
  --batch-size 16 \
  --output outputs/adv_sent_0_eval_bert_base.csv \
  > logs/eval_sent_0_bert_base.log 2>&1 &
```

---

## 6. 实验三：分桶分析（可选）
`exp3_analysis.py` 需要带 `pred` 的 clean/adv 预测文件：
- 可用 `victim_llm.py` 或 `victim_bert.py` 先得到 `pred` 列。
- 运行示例：按 `interaction_strategy` 分桶 ASR
```bash
python bin/exp3_analysis.py \
  --clean outputs/clean_pred.csv \
  --adv outputs/adv_sent_0_pred.csv \
  --bucket interaction_strategy \
  --output outputs/results_exp3.csv
```
