import re
import string
from typing import List, Dict, Any
import difflib
from bs4 import BeautifulSoup

# 工具函数

# 数据集加载（streaming 模式 + 采样）
# datasets = {
    # load_dataset("google-research-datasets/natural_questions", None, "validation"),
    # load_dataset("mandarjoshi/trivia_qa", "rc", "validation"),
    # load_dataset("rajpurkar/squad", None, "validation"),
    # load_dataset("stanfordnlp/web_questions", None, "test"),
    # load_dataset("cais/mmlu", "all", "validation"),
    # load_dataset("wics/strategy-qa", None, split="test"),
    # load_dataset("hotpot_qa", "distractor", "validation", trust_remote_code=True)
# }
def get_question_answer(dataset_name, sample):
    # 提取question和gold answers
    if dataset_name == "trivia_qa":
        return get_trivia_qa(sample)
    elif dataset_name == "natural_questions":
        return get_natural_questions(sample)
    elif dataset_name == "strategy_qa":
        return get_strategyqa(sample)
    elif dataset_name =="mmlu":
        return get_mmlu(sample)
    elif dataset_name == "web_questions":
        return get_web_questions(sample)
    else:
        return get_squad(sample)

def normalize_answer(s: str) -> str:
    """
    参考 SQuAD 官方评测脚本实现：
        转成小写
        去除英文冠词（a, an, the）
        去除标点符号
        合并多余空格
    """

    def lower(text: str) -> str:
        return text.lower()

    def remove_articles(text: str) -> str:
        # \b 表示单词边界，确保只去掉独立的 a/an/the
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def remove_punc(text: str) -> str:
        # 利用 string.punctuation 列表去除所有英文标点
        return ''.join(ch for ch in text if ch not in set(string.punctuation))

    def white_space_fix(text: str) -> str:
        # 将多个空白字符合并为一个，并去掉两端空格
        return ' '.join(text.split())

    # 按顺序执行各步
    text = s
    text = lower(text)
    text = remove_articles(text)
    text = remove_punc(text)
    text = white_space_fix(text)
    return text

def compute_score(answer: str, gold_list: list[str]):
    """
    对每个 gold answer 计算 token 级别的 P/R/F1，返回最高的那组值
    """
    # 将参考答案统一为列表形式
    if isinstance(gold_list, str):
        gold_list = [gold_list]
    if gold_list == []:
        if answer in ["", "I don’t know", "no answer", "not found", "I don’t know.", "n/a"]: 
            return 1.0, 1.0, 1.0
        else:
            return 0.0, 0.0, 0.0
    
    # 归一化预测答案（小写化、去标点和冠词等）
    pred_norm = normalize_answer(answer)
    gold_norms = [normalize_answer(g) for g in gold_list]
    
    # 检测是否为选择题
    is_choice = False
    # 单字符选项（如 'A','B','C'）
    if all(re.fullmatch(r"[a-z0-9]", g) for g in gold_norms):
        is_choice = True
    # 二元选项（如 'true','false'）
    if set(gold_norms) <= {"true", "false"}:
        is_choice = True

    if is_choice:
        # 任一参考答案相同即视为完全正确
        if pred_norm in gold_norms:
            return 1.0, 1.0, 1.0
        else:
            return 0.0, 0.0, 0.0
    
    # 否则则是简答题
    pred_tokens = pred_norm.split()
    if not pred_tokens:
        return 0.0, 0.0, 0.0
    best_p = best_r = best_f1 = 0.0

    for g in gold_norms:
        gold_tokens = normalize_answer(g).split()
        if len(gold_tokens) == 0:
            continue

        common_tokens = len(set(pred_tokens) & set(gold_tokens))
        if common_tokens == 0:
            p = r = f1 = 0
        else:
            p = common_tokens / len(pred_tokens)
            r = common_tokens / len(gold_tokens)
            f1 = 2 * p * r / (p + r)
        best_p = max(best_p, p)
        best_r = max(best_r, r)
        best_f1 = max(best_f1, f1)

    return best_p, best_r, best_f1

def strip_html(html: str) -> str:
    """简单地把 HTML 去标签，保留可读文字。"""
    return BeautifulSoup(html, "html.parser").get_text(separator=" ")

def get_natural_questions(sample):
    '''
    从 Natural Questions 数据集中提取问题和答案。
    '''
    question = sample["question"]["text"]
    print(f"Processing question: {question}")

    # 这里直接用整个 HTML 内容去标签后的文本
    html = sample["document"]["html"]
    background = f"Read the following context carefully. \
        Do not explain your answer or include any additional text, \
            answer the question using **only** a span (exact phrase) from the context.\n Context: {strip_html(html)}" 
    # print(f"Question background: {background}")

    # 提取 short_answers
    ann = sample["annotations"]
    short_ans = ann.get("short_answers", [])

    gold_answers = []
    if short_ans:
        for sa in short_ans:
            gold_answers.extend(sa["text"])
    else:
        # 没有 short_answers 时 fallback 到 long_answer
        long_ans = ann.get("long_answer", {})
        if long_ans and long_ans.get("start_byte", -1) >= 0:
            start, end = long_ans ["start_byte"], long_ans ["end_byte"]
            gold_answers = [strip_html(html[start:end])]
    print(f"Gold answers: {gold_answers}")
    return background, question, gold_answers

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

    print(f"Gold answers: {gold_answers}")
    background = ""
    return background, question, gold_answers

def get_squad(sample):
    '''
    从 SQuAD 数据集中提取问题和答案。
    hotqa也可以使用这个方法
    '''
    question = sample["question"]
    print(f"Processing question: {question}")

    context = sample.get('context')
    background = f"Read the following context carefully. \
        Do not explain your answer or include any additional text, \
            answer the question using **only** a span (exact phrase) from the context.\n Context: {context}" 
    print(f"Question background: {background}")
    
    gold_answers = sample["answers"].get('text', [])
    print(f"Gold answers: {gold_answers}")
    return background, question, gold_answers

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
    background = ""
    return background, question, gold_answers

def get_mmlu(sample):
    ''' 
    从 MMLU 数据集中提取问题和答案。
    '''
    question = sample["question"]
    print(f"Processing question: {question}")

    choices = sample["choices"]
    options = "\n".join([f"{i}. {c}" for i, c in enumerate(choices)])
    background = f"Choices:\n{options}\n\nWhich one is correct?\nThis is a multiple-choice question. You must choose the correct answer from the options 1, 2, 3, or 4. \
    Answer with only the number of the correct choice. Do not explain your answer or include any additional text. Just reply with: 1, 2, 3, or 4"
    
    print(f"Question background: {background}")
    
    gold_answers = sample["answer"]
    print(f"Gold answers: {gold_answers}")
    return background, question, str(gold_answers)

def get_strategyqa(sample):
    # 拼接 question + description
    question = sample.get("question", "").strip()
    desc = sample.get("description", "").strip()
    background = f"This is a true or false question. Please answer with either True or False. Do not provide any explanation or extra words. \
    Only reply with: True or False\n Background: {desc}\n"
    print(f"Processing question: {question}")
    print(f"Question background: {background}")

    gold_answer = sample["answer"]
    print(f"Gold answer: {gold_answer}")
    return background, question, str(gold_answer)

def exact_match(ans: str, gold_ans: List[str]) -> bool:
    ans_norm = normalize_answer(ans)
    return any(ans_norm in normalize_answer(g) for g in gold_ans)

# === 语义匹配函数（基于简单字符串相似度，实际部署可替换为向量相似度）===
def semantic_match(ans: str, gold_ans: List[str], threshold: float = 0.85) -> bool:
    ans_norm = normalize_answer(ans)
    for g in gold_ans:
        g_norm = normalize_answer(g)
        ratio = difflib.SequenceMatcher(None, ans_norm, g_norm).ratio()
        if ratio >= threshold:
            return True
    return False

def compute_hit(answer:str, gold_answer:List[str], context:str, contexts:List[str], threshold: float = 0.85):
    hit = exact_match(answer, gold_answer)
    if not hit:
        hit = semantic_match(answer, gold_answer, threshold)

    em = exact_match(context, contexts)
    return hit, em
