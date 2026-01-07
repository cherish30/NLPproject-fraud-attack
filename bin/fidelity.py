"""
保真度检测：规则层 + 语义层。
"""
from __future__ import annotations

import re
from typing import Dict, Tuple

import numpy as np

try:
    from sentence_transformers import SentenceTransformer
    _HAS_ST = True
except Exception:
    _HAS_ST = False
    SentenceTransformer = None  # type: ignore


def extract_sensitive(text: str) -> set[str]:
    """提取数字串/URL/邮箱等关键信息。"""
    if not isinstance(text, str):
        return set()
    patterns = [
        r"\d{6,}",  # 长数字串
        r"https?://[^\s]+",
        r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}",
    ]
    items = set()
    for pat in patterns:
        items.update(re.findall(pat, text))
    return items


def rule_consistency(a: str, b: str) -> bool:
    return extract_sensitive(a) == extract_sensitive(b)


_st_model = None


def get_st_model():
    global _st_model
    if _st_model is None and _HAS_ST:
        _st_model = SentenceTransformer("text2vec-base-chinese")
    return _st_model


def semantic_similarity(a: str, b: str) -> float:
    if not _HAS_ST:
        # 退化版：字符级 Jaccard
        set_a, set_b = set(a), set(b)
        inter = len(set_a & set_b)
        union = len(set_a | set_b) or 1
        return inter / union
    model = get_st_model()
    if model is None:
        return 0.0
    emb = model.encode([a, b], normalize_embeddings=True)
    return float(np.dot(emb[0], emb[1]))


def check_fidelity(x_clean: str, x_adv: str, sim_threshold: float = 0.6) -> Tuple[bool, Dict[str, float]]:
    """返回 (通过与否, 细节)"""
    rule_pass = rule_consistency(x_clean, x_adv)
    sim = semantic_similarity(x_clean, x_adv)
    passed = rule_pass and sim >= sim_threshold
    return passed, {"rule_pass": float(rule_pass), "semantic_sim": sim}
