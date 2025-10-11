import re
import string
from typing import List, Dict, Any
import difflib
from bs4 import BeautifulSoup
import yaml

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

with open("prompt.yaml", "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

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
    elif dataset_name == "squad":
        return get_squad(sample)
    elif dataset_name == "hot_qa":
        return get_hot_pot(sample)

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
        if answer in ["", "I don't know", "no answer", "not found", "I don't know.", "None", "none"]: 
            return 1.0, 1.0, 1.0
        else:
            return 0.0, 0.0, 0.0
    
    # 归一化预测答案（小写化、去标点和冠词等）
    pred_norm = normalize_answer(answer)
    gold_norms = [normalize_answer(g) for g in gold_list]
    
    # 检测是否为选择题
    is_choice = False
    # 单字符选项（如 'A','B','C'）
    if all(re.fullmatch(r"[a-d1-4]", g) for g in gold_norms):
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
        gold_tokens = g.split()
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
    """去除 HTML 标签，清理多余换行和空格"""
    # 提取纯文本，先用换行作为分隔
    text = BeautifulSoup(html, "html.parser").get_text(separator=" ")
    # 去掉 " 空格 + 换行 " 的情况
    text = re.sub(r" +\n", "\n", text)
    # 把多个连续换行压缩成一个
    text = re.sub(r"\n+", "\n", text)
    # 去掉首尾空白
    return text.strip()

def get_natural_questions(sample):
    '''
    从 Natural Questions 数据集中提取问题和答案。
    '''
    question = sample["question"]["text"]
    # print(f"Processing question: {question}")

    # 这里直接用整个 HTML 内容去标签后的文本
    html = sample["document"]["html"]
    background = {}
    template = config["fewshots"]["natural_questions"]["instruction"]["template"]
    background["Instruction"] = template
    background["fewshot"] = config["fewshots"]["natural_questions"]["examples"]["content"]
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
        long_ans = ann.get("long_answer", [])
        for la in long_ans:
            if la.get("candidate_index", -1) >= 0 and la.get("start_byte", -1) >= 0:
                start, end = la ["start_byte"], la ["end_byte"]
                gold_answers.append(strip_html(html[start:end]))
    if len(gold_answers) == 0:
        yes_no_answer = ann.get("yes_no_answer", [])
        for g in yes_no_answer:
            if g == 1:
                gold_answers.append("true")
            elif g == 0:
                gold_answers.append("false")
    # print(f"Gold answers: {gold_answers}")
    return background, question, gold_answers

def get_trivia_qa(sample):
    '''
    从 Trivia QA 数据集中提取问题和答案。
    '''
    if "question" in sample:
        question = sample["question"]
    else:
        raise KeyError("无法在样本中找到 question 字段")
    # print(f"Processing question: {question}")
    background = {}
    template = config["fewshots"]["trivia_qa"]["instruction"]["template"]
    background["Instruction"] = template
    background["fewshot"] = config["fewshots"]["trivia_qa"]["examples"]["content"]
    
    # answer 可能是字符串，也可能是list
    if "answer" in sample:
        gold_answers = sample["answer"].get('aliases', [])

    # print(f"Gold answers: {gold_answers}")
    return background, question, gold_answers

def get_squad(sample):
    '''
    从 SQuAD 数据集中提取问题和答案。
    '''
    question = sample["question"]
    # print(f"Processing question: {question}")

    context = sample.get('context')
    background = {}
    template = config["fewshots"]["squad"]["instruction"]["template"]
    background["Instruction"] = template
    background["fewshot"] = config["fewshots"]["squad"]["examples"]["content"]
    # print(f"Question background: {background}")
    
    gold_answers = sample["answers"].get('text', [])
    # print(f"Gold answers: {gold_answers}")
    return background, question, gold_answers

def get_hot_pot(sample):
    question = sample["question"]
    # print(f"Processing question: {question}")

    context = sample.get('context')
    background = {}
    template = config["fewshots"]["hot_qa"]["instruction"]["template"]
    background["Instruction"] = template
    background["fewshot"] = config["fewshots"]["hot_qa"]["examples"]["content"]
    # print(f"Question background: {background}")
    
    gold_answers = sample["answer"]
    # print(f"Gold answers: {gold_answers}")
    return background, question, gold_answers

def get_web_questions(sample):
    '''
    从 Web Questions 数据集中提取问题和答案。
    '''
    if "question" in sample:
        question = sample["question"]
    else:
        raise KeyError("无法在样本中找到 question 字段")
    # print(f"Processing question: {question}")
    background = {}
    template = config["fewshots"]["web_questions"]["instruction"]["template"]
    background["Instruction"] = template
    background["fewshot"] = config["fewshots"]["web_questions"]["examples"]["content"]

    if "answer" in sample:
        gold_answers = sample["answer"]
    elif "answers" in sample:
        gold_answers = sample["answers"]
    else:
        gold_answers = []
    # print(f"Gold answers: {gold_answers}")
    return background, question, gold_answers

def get_mmlu(sample):
    ''' 
    从 MMLU 数据集中提取问题和答案。
    '''
    question = sample["question"]
    # print(f"Processing question: {question}")

    choices = sample["choices"]
    options = "\n".join([f"{i}. {c}" for i, c in enumerate(choices)])
    background = {}
    template = config["fewshots"]["mmlu"]["instruction"]["template"]
    background["Instruction"] = template.format(options=options)
    background["fewshot"] = config["fewshots"]["mmlu"]["examples"]["content"]
    
    # print(f"Question background: {background}")
    
    gold_answers = sample["answer"]
    # print(f"Gold answers: {gold_answers}")
    return background, question, str(gold_answers)

def get_strategyqa(sample):
    # 拼接 question + description
    question = sample.get("question", "").strip()
    desc = sample.get("description", "").strip()
    background = {}
    template = config["fewshots"]["strategy_qa"]["instruction"]["template"]
    background["Instruction"] = template
    background["fewshot"] = config["fewshots"]["strategy_qa"]["examples"]["content"]
    
    # print(f"Processing question: {question}")
    # print(f"Question background: {background}")

    gold_answer = sample["answer"]
    # print(f"Gold answer: {gold_answer}")
    return background, question, str(gold_answer)

def get_single_humanqa(sample):
    # 拼接 question + description
    question = sample.get("question", "").strip()
    if question == "":
        question = sample.get("question_en", "").strip()
    background = {}
    template = config["fewshots"]["single_domain_human_qa"]["instruction"]["template"]
    background["Instruction"] = template
    background["fewshot"] = config["fewshots"]["single_domain_human_qa"]["examples"]["content"]

    gold_answer = sample.get("answer", [])
    if gold_answer == []:
        gold_answer = sample.get("answer_en", [])
    return background, question, gold_answer

def get_cross_humanqa(sample):
    # 拼接 question + description
    question = sample.get("question", "").strip()
    background = {}
    template = config["fewshots"]["cross_domain_human_qa"]["instruction"]["template"]
    background["Instruction"] = template
    background["fewshot"] = config["fewshots"]["cross_domain_human_qa"]["examples"]["content"]

    gold_answer = sample["answer"]
    return background, question, str(gold_answer)

def exact_match(ans: str, gold_ans: List[str]) -> bool:
    ans_norm = normalize_answer(ans)
    return any(normalize_answer(g) in ans_norm for g in gold_ans)

# === 语义匹配函数（基于简单字符串相似度）===
def semantic_match(ans: str, gold_ans: List[str], threshold: float = 0.85) -> bool:
    ans_norm = normalize_answer(ans)
    return any(difflib.SequenceMatcher(None, ans_norm, normalize_answer(g)).ratio() >= threshold
               for g in gold_ans)

def compute_hit(answer:str, gold_answer:List[str], retrieval:list[str], contexts, threshold: float = 0.85):
    # 计算 EM
    em = 1 if exact_match(answer, gold_answer) else 0
    
    # 计算 context 命中率（以 recall 为例）
    matched = 0
    for c in retrieval:
        if exact_match(c, contexts) or semantic_match(c, contexts, threshold):
            matched += 1
    if len(contexts) > 1:
        hit = matched / len(retrieval) if len(retrieval) > 0 else 0
    elif len(contexts) == 1:
        hit = 1 if matched > 0 else 0
    return hit, em
