from datasets import load_dataset
from evaluate import load
import time
import pandas as pd
from client import Client
from server import Server

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
    return sampled

ds = load_dataset("google-research-datasets/natural_questions",
                      split="validation", streaming=True)

# 加载指标
metric_qa = load("squad")

def evaluate_natural_questions( 
        ds,
        clients: list[Client],
        llm_server: Server,
        sample_limit=100,
        top_k=5,
        output_csv: str = "natural_questions_results.csv"): 
    samples = []
    for i, ex in enumerate(ds):
        if i >= sample_limit:
            break
        samples.append(ex)
    print(f"Loaded {len(samples)} Natural Questions samples.")

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
        answer = llm_server.generate_answer(question, contexts)
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
    llm_server = Server(model_name="qwen3:4b")
    evaluate_natural_questions(ds, clients, llm_server, sample_limit=100, top_k=5, output_csv="natural_questions_results.csv")
