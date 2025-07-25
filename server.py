from typing import List, Tuple, Dict
from langchain.chat_models import ChatOllama
from langchain.schema import Document
from client import Client
import time
import re


class Server:
    """
    Server 类，负责：
    1) 接收客户端选择的上下文数据
    2) 验证数据完整性（通过 Proof 信息）
    3) 调用 Ollama 部署的 Qwen3:4B 模型生成答案
    """
    def __init__(self, model_name: str = "qwen3:4b"):
        self.llm = ChatOllama(model=model_name)

    def verify_documents(self):
        return

    def build_prompt(self, query: str, contexts: List[str]) -> str:
        """构造 Prompt，将 query 和上下文拼接"""
        prompt = "You are an AI assistant. Use the following contexts to answer the question:\n"
        for i, c in enumerate(contexts, 1):
            prompt += f"Context {i}: {c}\n"
        prompt += f"Question: {query}\nAnswer:"
        return prompt

    def generate_answer(self, query: str, contexts: List[str]) -> str:
        """
        验证 Proof 后调用模型生成答案
        """
        # if not self.verify_documents(contexts, proofs):
        #     raise ValueError("Proof verification failed! Data may be tampered.")
        prompt = self.build_prompt(query, contexts)
        response = self.llm.predict(prompt)
        return response
    
    def clean_answer(self, raw: str):
        """
        去除 <think> 标签和其中的内容，并去掉多余空白
        """
        # 去掉所有 <think>…</think> 区段
        cleaned = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL)
        # 去掉多余的换行和空格
        return cleaned.strip()

    def multi_client_generate(self, query:str, clients: List[Client], top_k=5):
        """
        多客户端检索，返回答案和相关文档
        """
        # RAG 检索 + LLM 生成 
        start = time.time()
        all_proofs = []
        for c in clients:
            proofs, q_vec = c.retrieve(query, top_k)
            all_proofs.extend(proofs)
            
        # 根据得分进行排序，选出最优proofs
        all_proofs.sort(key=lambda p: p.score, reverse=True)

        # 请求对应client提供真实上下文
        contexts = [r.document.page_content for r in all_proofs[:top_k]]
        
        #生成答案并清洗
        raw_answer = self.generate_answer(query, contexts)
        answer = self.clean_answer(raw_answer)

        latency = time.time() - start

        return latency, answer
    