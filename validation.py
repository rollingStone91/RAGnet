from evaluate import load as load_metric
import pandas as pd
from client import Client
from server import Server
import validation_tools
from server_algo import Server_with_Algorithm, Cost_Algorithm

def evaluate_datasets(clients: list[Client], server: Server, top_k=5, samples=[],
                                output_csv: str = "trivia_qa_results.csv", dataset_name:str = "trivia_qa"): 
    """
    clients: 创建的多个client
    server: 用于生成的server
    samples: 先加载数据集为dataset
    output_csv: 保存的文件名
    dataset_name: 数据集的名称("trivia_qa", "natural_questions", "squad", "mmlu",
                             "strategy_qa", "web_questions", "hot_qa")
    """
    results = []
    for idx, sample in enumerate(samples):
        background, question, gold_answers = validation_tools.get_question_answer(dataset_name, sample)

        # 调用 LLM Server
        retrieve_latency, generate_latency, contexts, answer = server.multi_client_generate(background, question, clients, top_k)

        # 计算自定义 P/R/F1  
        precision, recall, f1_score = validation_tools.compute_score(answer, gold_answers)

        results.append({
            "question": question,
            "gold_answers": gold_answers,
            "answer": answer,
            "precision": precision,
            "recall": recall,
            "f1_score": f1_score,
            "retrieve_latency": retrieve_latency,
            "generate_latency": generate_latency
        })

        if idx % 10 == 0:
            print(f"[{idx}] F1={f1_score:.2f}, P={precision:.2f}, R={recall:.2f}")


    # 保存到 CSV
    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False, encoding="utf-8-sig")
    print(f"Saved Natural Questions results to {output_csv}")

def datasets_costs(clients: list[Client], server: Server_with_Algorithm, top_k=5, samples=[],
                                output_csv: str = "trivia_qa_costs.csv"): 
    """
    clients: 创建的多个client
    server: 用于生成的server
    samples: 先加载数据集为dataset
    output_csv: 保存的文件名
    dataset_name: 数据集的名称("trivia_qa", "natural_questions", "squad", "mmlu",
                             "strategy_qa", "web_questions", "hot_qa")
    """
    costs = []
    for idx, sample in enumerate(samples):
        question = sample["question"]
        background = f"Generate exactly 3 distinct answers that can all be verified from the document and\
the answers MUST be directly found or clearly derivable from the provided document content.\n Context: {sample['context']}"
        
        print(f"Processing question: {question}")
        print(f"Background: {sample['context']}")
        print(f"Gold_Answer: {sample['answer']}")
        # 调用 LLM Server
        cost, answer = server.multi_client_generate(background, question, clients, top_k)

        costs.append({
            "idx": idx,
            "retrieval_time": cost.retrieval_time,
            "por_proof_time": cost.por_proof_time,
            "por_verify_time": cost.por_verify_time,
            "por_proof_size": cost.por_proof_size,
            "generation_time": cost.generation_time,
            "pog_proof_time": cost.pog_proof_time,
            "pog_verify_time": cost.pog_verify_time,
            "pog_proof_size": cost.pog_proof_size,
        })

        if idx % 10 == 0:
            print(f"[{idx}] retrieval_time={cost.retrieval_time:.5f}, por_proof_time={cost.por_proof_time:.5f}, \
                  por_verify_time={cost.por_verify_time:.5f}, por_proof_size={cost.por_proof_size:.5f}, \
                    generation_time={cost.generation_time:.5f}, pog_proof_time={cost.pog_proof_time:.5f}, \
                        pog_verify_time={cost.pog_verify_time:.5f}, pog_proof_size={cost.pog_proof_size:.5f}")
    # 保存到 CSV
    df = pd.DataFrame(costs)
    df.to_csv(output_csv, index=False, encoding="utf-8-sig")
    print(f"Saved Natural Questions results to {output_csv}")

def evaluate_hit_rate(clients: list[Client], server: Server, top_k=5, samples=[], similarity_threshold=0.85,
                      output_csv: str = "semi_professional_baseline_topk_5.csv"):
    hit_rates = []
    for idx, sample in enumerate(samples):
        question = sample["question"]
        background = f"Generate exactly 3 distinct answers that can all be verified from the document and\
the answers MUST be directly found or clearly derivable from the provided document content.\n Context: {sample['context']}"
        
        print(f"Processing question: {question}")
        print(f"Background: {sample['context']}")
        print(f"Gold_Answer: {sample['answer']}")

        # 调用 LLM Server
        retrieve_latency, generate_latency, contexts, answer = server.multi_client_generate(background, question, clients, top_k)

        hit, exact_match = validation_tools.compute_hit(answer, sample['answer'], sample['context'], contexts, similarity_threshold)

        hit_rates.append({
            "idx": idx,
            "question_id": sample['id'],
            "hit": hit,
            "exact_match": exact_match,
            "retrieve_latency": retrieve_latency,
            "generate_latency": generate_latency
        })

        if idx % 10 == 0:
            print(f"[{idx}] hit={hit}, exact_match={exact_match}")

    # 保存到 CSV
    df = pd.DataFrame(hit_rates)
    df.to_csv(output_csv, index=False, encoding="utf-8-sig")
    print(f"Saved Natural Questions results to {output_csv}")