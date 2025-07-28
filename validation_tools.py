from datasets import load_dataset
import re

# 工具函数

# 数据集加载（streaming 模式 + 采样）
# datasets = {
    # load_dataset("google-research-datasets/natural_questions", None, "validation"),
    # load_dataset("mandarjoshi/trivia_qa", "rc", "validation"),
    # load_dataset("rajpurkar/squad", None, "validation"),
    # load_dataset("stanfordnlp/web_questions", None, "test"),
    # load_dataset("cais/mmlu", "all", "validation"),
    # load_dataset("wics/strategy-qa", None, split="train"),
    # load_dataset("hotpot_qa", "distractor", "validation", trust_remote_code=True)
# }
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

def normalize_text(s: str):
    """小写，去标点，去多余空格"""
    s = s.lower()
    s = re.sub(r"[^a-z0-9\s]", "", s)
    return " ".join(s.split())

def compute_prf(pred: str, golds: list[str]):
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

def get_natural_questions(sample):
    '''
    从 Natural Questions 数据集中提取问题和答案。
    '''
    question = sample["question"]["text"]
    print(f"Processing question: {question}")
    gold_answers = sample.get("answers", {}).get("text", [])
    print(f"Gold answers: {gold_answers}")
    if not gold_answers:
        gold_answers = [""]  # 占位
    return question, gold_answers

def get_trivia_qa(sample):
    '''
    从 Trivia QA 数据集中提取问题和答案。
    '''
    if "question" in sample:
        question = sample["question"]
    elif "query" in sample:
        question = sample["query"]
    else:
        raise KeyError("无法在样本中找到 question 字段")
    print(f"Processing question: {question}")

    # answer 可能是字符串，也可能是list
    if "answer" in sample:
        gold_answers = sample["answer"].get('aliases', [])
    elif "answers" in sample:
        gold_answers = sample["answers"].get('aliases', [])
    else:
        gold_answers = []
    print(f"Gold answers: {gold_answers}")
    return question, gold_answers

def get_squad(sample):
    '''
    从 SQuAD 数据集中提取问题和答案。
    '''
    if "question" in sample:
        question = sample["question"]
    else:
        raise KeyError("无法在样本中找到 question 字段")
    print(f"Processing question: {question}")

    if "answer" in sample:
        gold_answers = sample["answer"].get('text', [])
    elif "answers" in sample:
        gold_answers = sample["answers"].get('text', [])
    else:
        gold_answers = []
    print(f"Gold answers: {gold_answers}")
    return question, gold_answers

def get_web_questions(sample):
    '''
    从 Web Questions 数据集中提取问题和答案。
    '''
    if "question" in sample:
        question = sample["question"]
    else:
        raise KeyError("无法在样本中找到 question 字段")
    print(f"Processing question: {question}")

    if "answer" in sample:
        gold_answers = sample["answer"]
    elif "answers" in sample:
        gold_answers = sample["answers"]
    else:
        gold_answers = []
    print(f"Gold answers: {gold_answers}")
    return question, gold_answers

def get_mmlu(sample):
    ''' 
    从 MMLU 数据集中提取问题和答案。
    '''
    if "question" in sample:
        question = sample["question"]
    else:
        raise KeyError("无法在样本中找到 question 字段")
    choices = sample["choices"]
    options = "\n".join([f"{chr(65+i)}. {c}" for i, c in enumerate(choices)])

    # 拼接
    query = f"{question}\n\nChoices:\n{options}\n\nWhich one is correct?"
    print(f"Processing question: {query}")

    if "answer" in sample:
        gold_answers = sample["answer"]
    elif "answers" in sample:
        gold_answers = sample["answers"]
    else:
        gold_answers = []
    print(f"Gold answers: {gold_answers}")
    return query, gold_answers