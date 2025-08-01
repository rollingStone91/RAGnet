from typing import List, Tuple, Dict
from langchain.chat_models import ChatOllama
from langchain.schema import Document
from client import Client
import asyncio
import time
import re
from privacy_proof import PrivacyProofAPI
import json
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.schema import HumanMessage, SystemMessage
import numpy as np

class Cost_Algorithm:
    def __init__(self, start):
        self.start = start
        self.retrieval_time = 0
        self.por_proof_time = 0
        self.por_verify_time = 0 
        self.por_proof_size = 0
        self.generation_time = 0
        self.pog_proof_time = 0
        self.pog_verify_time = 0
        self.pog_proof_size = 0


class Server_with_Algorithm:
    """
    Server 类，负责：
    1) 并行化多客户端检索
    2) 验证数据完整性（通过 Proof 信息）
    3) 调用 Ollama 部署的 Qwen3:4B 模型生成答案
    """
    def __init__(self, model_name: str = "qwen3:4b", base_url="http://2b895c11.r9.cpolar.top", 
                  model_path: str = "./models/qwen3-embedding-0.6b"):
        self.llm = ChatOllama(model=model_name)
        self.proof_api = PrivacyProofAPI(base_url=base_url)  # Optional: PrivacyProofAPI 实例
        self.embeddings = HuggingFaceEmbeddings(model_name=model_path,
                                                model_kwargs={"device": "cuda"},
                                                encode_kwargs={"normalize_embeddings": True},)
                                                # multi_process=True)

    def _retrieve_from_clients(self, clients, query: str, top_k: int):
        proofs = []
        q_vec = []
        for client in clients:
            p, q = client.retrieve(query, top_k)
            proofs.extend(p)
            q_vec.append(q)
            # 获取 FAISS 原始 index
            index = client.db.index
            # 查看向量维度
            print("原始向量维度:", index.d)
        self.cost.retrieval_time = time.time() - self.cost.start

        # 使用pedersen算法
        for p in proofs:
            # print(p.vector)
            # print(len(q_vec[0]))
            # print(f"Context: {p.document.page_content}")
            response = self.proof_api.gen_pedersen_proof(name="commonsense",
                                                            K=p.vector, Q=q_vec[0], 
                                                            data=p.document.page_content)
            print(f"gen pedersen proof:{response}")
            # response = json.loads(response)
            p.pedersen_id = response["proof_id"]
            self.cost.por_proof_size += response["space_cost"]
            self.cost.por_proof_time += response["time_cost"]

        return proofs, q_vec[0]

    def build_prompt(self, background:str, query: str, contexts: List[str], metadatas) -> str:
        """构造 Prompt，将 query 和上下文拼接"""
        # 精炼指令：System + User 模式
        system_msg = """
        You are a knowledgeable AI assistant. 
        - Use ONLY the following contexts and metadata to answer; do NOT use any outside knowledge.
        - If the answer cannot be found in the contexts, reply exactly “I don’t know.”
        - Provide standardized answers based on the "descriptions" of the question. 
        - Before the final answer, show your step-by-step reasoning prefixed with “<think>” and suffixed with “</think>”.
        """
        
        # Few‑Shot example
        few_shot = """
        ### Example
        Question: "What is at the center of our Solar System?"
        Next, here are the relevant contexts and metadata:
        Context 1: "The sun is the star at the center of the Solar System."
        Metadata 1: {...}
        Context 2: "A light-year is the distance light travels in one Julian year."
        Metadata 2: {...}
        <think>
        1. Identify the context that mentions "center of the Solar System".
        2. Context 1 states that the Sun is the star at the center.
        3. Context 2 is not relevant to this question.
        </think>
        Answer: The Sun.
        """

        user_msg = f"Question: {query}\n"
        if background:
            user_msg += f'Here is a description of the question: {background}\n'
        user_msg += f'Next, here are the relevant contexts and metadata:\n'
        for i, (c, m) in enumerate(zip(contexts, metadatas)):
            user_msg += f"[Context {i+1}] {c}\n[Metadata {i+1}] {json.dumps(m, ensure_ascii=False)}\n"
        human_msg = few_shot+"\n"+user_msg
        
        prompt = [SystemMessage(content=system_msg), HumanMessage(content=human_msg)]
        return prompt

    def clean_answer(self, raw: str):
        """
        去除 <think> 标签和其中的内容，并去掉多余空白
        """
        # 去掉所有 <think>…</think> 区段
        print(f"uncleaned answer: {raw}")
        cleaned = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL)
        # print(f"cleaned answer: {cleaned}")

        # 从 LLM 回复中提取 'Final Answer' 后的内容
        match = re.search(r"Answer[:\s]*(.*)", cleaned, flags=re.IGNORECASE) 
        if match:
            return match.group(1).strip()
        return cleaned.strip()
    
    def generate_answer(self, background:str, query: str, q_vec, proofs) -> str:
        contexts = []
        metadatas = []
        # 验证proof
        for p in proofs:
            # 取出文本并embed
            data = p.document.page_content
            # embed context
            # k = np.array(self.embeddings.embed_documents([data])[0], dtype=np.float32)  
            # k_vec=k.tolist()
            # print(len(k_vec))
            # 生成pogid
            res = self.proof_api.gen_pog(q_vec, p.vector, data)
            # res = json.loads(res)
            p.pog_id = res["proof_id"]
            self.cost.pog_proof_time += res["time_cost"]
            self.cost.pog_proof_size += res["space_cost"]

            # 验证proof
            msg = self.proof_api.verify_pog(p.pedersen_id, p.pog_id)
            # msg = json.loads(msg)
            print(f"verify pog:{msg}")
            
            # 验证通过则加入列表
            if(msg['msg'] == "ok"):
                contexts.append(data)
                metadatas.append(p.document.metadata)
                self.cost.pog_verify_time += res["time_cost"]
        
        proof_len = len(contexts) if len(contexts) > 0 else 1
        # 计算平均值
        self.cost.pog_proof_time = self.cost.pog_proof_time / proof_len
        self.cost.pog_proof_size = self.cost.pog_proof_size / proof_len
        self.cost.pog_verify_time = self.cost.pog_verify_time / proof_len

        # print(f"contexts: {contexts}") 
        # print(f"metadatas: {metadatas}")

        prompt = self.build_prompt(background, query, contexts, metadatas)
        
        self.cost.start = time.time()
        response = self.llm.invoke(prompt)
        self.cost.generation_time = time.time() - self.cost.start

        answer = self.clean_answer(response.content)
        print(f"answer:{answer}")
        return answer
    
    def multi_client_generate(self, background:str, query:str, clients: List[Client], top_k=5):
        """
        多客户端检索，返回答案和相关文档
        """
        # RAG 检索 + LLM 生成 
        self.cost = Cost_Algorithm(time.time())

        all_proofs, q_vec = self._retrieve_from_clients(clients, query, top_k)

        # 计算平均时间
        proof_len = top_k * len(clients)
        self.cost.por_proof_size = self.cost.por_proof_size / proof_len
        self.cost.por_proof_time = self.cost.por_proof_size / proof_len
        
        # print(f"q_vecs: {q_vec}")

        verified_proof = []
        # flatten proofs
        for p in all_proofs:
            response = self.proof_api.verify_pedersen_proof(proof_id=p.pedersen_id)
            # response = json.loads(response)
            print(f"verify pedersen proof:{response}")
            if(response['msg'] == "ok"):
                verified_proof.append(p)
            self.cost.por_verify_time += response["time_cost"]
        self.cost.por_verify_time = self.cost.por_verify_time / proof_len

        # 根据得分进行排序，选出最优proofs
        verified_proof.sort(key=lambda p: getattr(p, 'score', 0), reverse=True)

        # 取 Top-K
        selected = verified_proof[:top_k]
        
        #生成答案并清洗
        answer = self.generate_answer(background=background, query=query, q_vec=q_vec, proofs=selected)

        return self.cost, answer
    