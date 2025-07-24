from datasets import load_dataset
import re

# -----------------------
# 工具函数
# -----------------------

# 数据集加载（streaming 模式 + 采样）
def load_sampled_dataset(name, config=None, split="validation", sample_size=100):
    try:
        ds = load_dataset(name, config, split=split, streaming=True)
    except Exception:
        # 如果 validation split 不存在，用 test
        ds = load_dataset(name, config, split="test", streaming=True)
    
    sampled = []
    for i, example in enumerate(ds):
        if i >= sample_size:
            break
        sampled.append(example)
    print(f"Loaded {len(sampled)} samples from {name} ({split})")
    return sampled

def clean_answer(raw: str) -> str:
    """
    去除 <think> 标签和其中的内容，并去掉多余空白
    """
    # 去掉所有 <think>…</think> 区段
    cleaned = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL)
    # 去掉多余的换行和空格
    return cleaned.strip()

def normalize_text(s: str) -> str:
    """小写，去标点，去多余空格"""
    s = s.lower()
    s = re.sub(r"[^a-z0-9\s]", "", s)
    return " ".join(s.split())

def compute_prf1(pred: str, golds: list[str]) -> tuple[float,float,float]:
    """
    对每个 gold answer 计算 token 级别的 P/R/F1，返回最高的那组值
    """
    pred_tokens = normalize_text(pred).split()
    best_p = best_r = best_f1 = 0.0

    for g in golds:
        gold_tokens = normalize_text(g).split()
        if not pred_tokens or not gold_tokens:
            continue
        common = len(set(pred_tokens) & set(gold_tokens))
        p = common / len(pred_tokens)
        r = common / len(gold_tokens)
        f1 = 2*p*r/(p+r) if (p+r)>0 else 0.0
        best_p = max(best_p, p)
        best_r = max(best_r, r)
        best_f1 = max(best_f1, f1)

    return best_p, best_r, best_f1