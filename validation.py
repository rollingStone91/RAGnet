from evaluate import load as load_metric
import pandas as pd
from client import Client
from server import Server
from validation_tools import load_sampled_dataset, compute_prf
import validation_tools

datasets = {
    "natural_questions": load_sampled_dataset("google-research-datasets/natural_questions", split="validation", sample_size=100),
    "trivia_qa": load_sampled_dataset("mandarjoshi/trivia_qa", "rc", split="validation", sample_size=100),
    "squad": load_sampled_dataset("rajpurkar/squad", split="validation", sample_size=100),
    "web_questions": load_sampled_dataset("stanfordnlp/web_questions", split="test", sample_size=100),
    "mmlu": load_sampled_dataset("cais/mmlu", "all", split="validation", sample_size=100),
    "strategy_qa": load_sampled_dataset("wics/strategy-qa", split="validation", sample_size=100),
    "hotpot_qa": load_sampled_dataset("hotpot_qa", split="validation", sample_size=100)  # 移除 fullwiki
}

def evaluate_natural_questions(clients: list[Client], server: Server, top_k=5, dataset_name = 'natural_questions',
                                output_csv: str = "natural_questions_results.csv"): 
    results = []
    samples = datasets.get(dataset_name)
    for idx, sample in enumerate(samples):
        # 提取question和gold answers
        question, gold_answers = validation_tools.get_natural_questions(sample)

        # 调用 LLM Server
        latency, answer = server.multi_client_generate(question, clients, top_k)

        # 构造 SQuAD 格式输入
        prediction = [{"id": str(idx), "prediction_text": answer}]
        reference  = [{"id": str(idx), "answers": {"text": gold_answers, "answer_start": [0]}}]

        # 计算 EM 和 F1
        # 加载指标
        metric_qa = load_metric("squad")
        qa_metrics = metric_qa.compute(predictions=prediction, references=reference)
        em_score = qa_metrics["exact_match"]
        f1_score = qa_metrics["f1"]

        # 计算自定义 P/R/F1  
        precision, recall, pr_f1 = compute_prf(answer, gold_answers)

        results.append({
            "question": question,
            "gold_answers": gold_answers,
            "answer": answer,
            "exact_match": em_score,
            "squad_f1": f1_score,
            "precision": precision,
            "recall": recall,
            "pr_f1": pr_f1, # 自定义的F1分数
            "latency": latency
        })

        if idx % 10 == 0:
            print(f"[{idx}] EM={em_score}, F1={f1_score:.2f}, P={precision:.2f}, R={recall:.2f}, latency={latency:.2f}s")

    # 保存到 CSV
    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False, encoding="utf-8-sig")
    print(f"Saved Natural Questions results to {output_csv}")

if __name__ == "__main__":
    clients = [Client(vectorstore_path="./common_sense_db"), Client(vectorstore_path="./computer_science_coding_related_db"),
               Client(vectorstore_path="./law_related_db"), Client(vectorstore_path="./medicine_related_db")]
    for c in clients:
        c.load_vectorstore()
    server = Server(model_name="qwen3:4b")
    evaluate_natural_questions(clients, server, top_k=5, dataset_name='natural_questions', output_csv="natural_questions_results.csv")
