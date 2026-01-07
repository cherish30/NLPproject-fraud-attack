#!/usr/bin/env bash
# 一键生成 LLM 预测 + 分桶 ASR（按 interaction_strategy）
set -e

DATA_DIR="outputs"

# 1) 生成 clean 预测
python bin/victim_llm.py --data ${DATA_DIR}/clean_test_sample400.csv --output ${DATA_DIR}/clean_pred_llm.csv

# 2) 对四个对抗集生成预测
for f in adv_word_0 adv_word_1 adv_sent_0 adv_sent_1; do
  python bin/victim_llm.py --data ${DATA_DIR}/${f}.csv --output ${DATA_DIR}/${f}_pred_llm.csv
done

# 3) 分桶 ASR（按 interaction_strategy），分别输出 results_exp3_llm_{name}.csv
for f in adv_word_0 adv_word_1 adv_sent_0 adv_sent_1; do
  python bin/exp3_analysis.py \
    --clean ${DATA_DIR}/clean_pred_llm.csv \
    --adv ${DATA_DIR}/${f}_pred_llm.csv \
    --bucket interaction_strategy \
    --output ${DATA_DIR}/results_exp3_llm_${f}.csv
done

echo "[info] 完成：clean/adv 预测与分桶 ASR 已输出到 ${DATA_DIR}/results_exp3_llm_*.csv"
