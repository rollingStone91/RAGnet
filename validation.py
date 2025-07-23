from datasets import load_dataset
from evaluate import load
import time
import pandas as pd
from client import Client
from server import Server

datasets = {
    "natural_questions": load_dataset("google-research-datasets/natural_questions", split="validation"),
    "trivia_qa": load_dataset("mandarjoshi/trivia_qa", "unfiltered", split="validation"),
    "squad": load_dataset("rajpurkar/squad", split="validation"),
    "web_questions": load_dataset("stanfordnlp/web_questions", split="test"),
    "mmlu": load_dataset("cais/mmlu", "all", split="validation"),
    "strategy_qa": load_dataset("wics/strategy-qa", split="validation"),
    "hotpot_qa": load_dataset("hotpot_qa", "fullwiki", split="validation")
}

# 加载指标
metric_em = load("exact_match")
metric_f1 = load("f1")
metric_rouge = load("rouge")

def evaluate_dataset(dataset, clients, llm_server, top_k=5):
    results = []
    for i, sample in enumerate(dataset):
        question = sample["question"]
        gold_answers = sample.get("answers", {}).get("text", [""])

        start_time = time.time()
        # 向所有client广播 query
        all_proofs = []
        for c in clients:
            proofs, q_vec = c.retrieve(question, top_k)
            all_proofs.extend(proofs)

        # 根据得分进行排序，选出最优proofs
        all_proofs.sort(key=lambda p: p.score, reverse=True)

        # 请求对应client提供真实上下文
        contexts = [r.document.page_content for r in all_proofs[:top_k]]

        # 调用 LLM Server
        answer = llm_server.generate_answer(question, contexts)
        latency = time.time() - start_time

        # 计算指标
        em_score = metric_em.compute(predictions=[answer], references=[gold_answers])["exact_match"]
        f1_score = metric_f1.compute(predictions=[answer], references=[gold_answers])["f1"]
        rouge_score = metric_rouge.compute(predictions=[answer], references=[gold_answers])["rougeL"]

        results.append({
            "question": question,
            "gold": gold_answers,
            "answer": answer,
            "em": em_score,
            "f1": f1_score,
            "rougeL": rouge_score,
            "latency": latency,
            "retrieved_docs": all_proofs[:top_k]
        })

        if i % 50 == 0:
            print(f"Processed {i} samples...")

    return results

if __name__ == "__main__":
    # 假设 client 和 llm_server 已经初始化
    clients = [Client(vectorstore_path="./common_sense_db"), Client("./computer_science_coding_related_db"),
               Client("./law_related_db"), Client("./medicine_related_db")]
    llm_server = Server(model_name="qwen3:4b")

    all_results = []
    for name, dataset in datasets.items():
        print(f"Evaluating {name} dataset...")
        results = evaluate_dataset(dataset, clients, llm_server)
        all_results.extend(results)

    # 保存结果到 CSV
    df = pd.DataFrame(results)
    df.to_csv("evaluation_results.csv", index=False)    
    print("Saving results to evaluation_results.csv...")