from evaluate import load
import time
import pandas as pd
from client import Client
from server import Server

from validation_tools import load_sampled_dataset

# datasets = {
#     "natural_questions": load_sampled_dataset("google-research-datasets/natural_questions", split="validation", sample_size=100),
#     "trivia_qa": load_sampled_dataset("mandarjoshi/trivia_qa", "unfiltered", split="validation", sample_size=100),
#     "squad": load_sampled_dataset("rajpurkar/squad", split="validation", sample_size=100),
#     "web_questions": load_sampled_dataset("stanfordnlp/web_questions", split="test", sample_size=100),
#     "mmlu": load_sampled_dataset("cais/mmlu", "all", split="validation", sample_size=100),
#     "strategy_qa": load_sampled_dataset("wics/strategy-qa", split="validation", sample_size=100),
#     "hotpot_qa": load_sampled_dataset("hotpot_qa", split="validation", sample_size=100)  # 移除 fullwiki
# }

samples = load_sampled_dataset("google-research-datasets/natural_questions",
                      split="validation")

# 加载指标
metric_qa = load("squad")

def evaluate_natural_questions(clients: list[Client], server: Server, top_k=5,
                                output_csv: str = "natural_questions_results.csv"): 
    results = []
    for idx, sample in enumerate(samples):
        question = sample["question"]["text"]
        print(f"Processing question: {question}")
        gold_answers = sample.get("answers", {}).get("text", [])
        print(f"Gold answers: {gold_answers}")
        if not gold_answers:
            gold_answers = [""]  # 占位

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
        answer = server.generate_answer(question, contexts)
        latency = time.time() - start_time

        # 构造 SQuAD 格式输入
        prediction = [{"id": str(idx), "prediction_text": answer}]
        reference  = [{
            "id": str(idx),
            "answers": {"text": gold_answers, "answer_start": [0]}
        }]

        # 计算 EM 和 F1
        qa_metrics = metric_qa.compute(predictions=prediction, references=reference)
        em_score = qa_metrics["exact_match"]
        f1_score = qa_metrics["f1"]

        results.append({
            "question": question,
            "answer": answer,
            "em": em_score,
            "f1": f1_score,
            "latency": latency,
            "retrieved_docs": contexts
        })

        if idx % 10 == 0:
            print(f"[{idx}] EM={em_score}, F1={f1_score}, latency={latency:.2f}s")

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
    evaluate_natural_questions(clients, server, top_k=5, output_csv="natural_questions_results.csv")
