from typing import List, Tuple, Dict
from langchain.chat_models import ChatOllama
from langchain.schema import Document
from client import Client
import asyncio
import time
import re
from privacy_proof import PrivacyProofAPI
import json
from langchain.schema import HumanMessage, SystemMessage


class Server:
    """
    Server 类，负责：
    1) 并行化多客户端检索
    2) 验证数据完整性（通过 Proof 信息）
    3) 调用 Ollama 部署的 Qwen3:4B 模型生成答案
    """
    def __init__(self, model_name: str = "qwen3:4b", base_url="http://439fdd8d.r16.vip.cpolar.cn"):
        self.llm = ChatOllama(model=model_name)
        self.proof_api = PrivacyProofAPI(base_url=base_url)  # Optional: PrivacyProofAPI 实例

    def _retrieve_from_clients(self, clients, query: str, top_k: int):
        proofs = []
        q_vec = []
        for client in clients:
            p, q = client.retrieve(query, top_k)
            proofs.extend(p)
            q_vec.append(q)
        return proofs, q_vec[0]

    def build_prompt(self, background:str, query: str, contexts: List[str], metadatas) -> str:
        """构造 Prompt，将 query 和上下文拼接"""
        # 精炼指令：System + User 模式
        system_msg = """
        You are a knowledgeable AI assistant. 
        - Use ONLY the following contexts and metadata to answer; do NOT use any outside knowledge.
        - The earlier a context appears, the more likely it contains the correct answer — prioritize earlier contexts when reasoning.
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
    
    def generate_answer(self, background:str, query: str, contexts: List[str], metadatas) -> str:
        prompt = self.build_prompt(background, query, contexts, metadatas)
        response = self.llm.invoke(prompt)
        answer = self.clean_answer(response.content)
        print(f"answer:{answer}")
        return answer
    
    def multi_client_generate(self, background:str, query:str, clients: List[Client], top_k=5, timeout: float = 60.0):
        """
        多客户端检索，返回答案和相关文档
        """
        # RAG 检索 + LLM 生成 
        start = time.time()

        all_proofs, q_vec = self._retrieve_from_clients(clients, query, top_k)
        
        retrieve_latency = time.time() - start

        # print(f"q_vec: {q_vec}")

            
        # 根据得分进行排序，选出最优proofs
        all_proofs.sort(key=lambda p: getattr(p, 'score', 0), reverse=True)

        # 取 Top-K
        selected = all_proofs[:top_k]
        # 请求对应client提供真实上下文
        contexts = [r.document.page_content for r in selected]
        metadatas = [r.document.metadata for r in selected]
        # scores = [r.score for r in selected]
        # print(f"contexts: {contexts}") 
        # print(f"metadatas: {metadatas}")
        # print(f"scores: {scores}")
        
        #生成答案并清洗
        start = time.time()
        answer = self.generate_answer(background, query, contexts, metadatas)
        generate_latency = time.time() - start

        return retrieve_latency, generate_latency, contexts, answer
    