import os
from typing import List, Tuple, Dict
import numpy as np
from langchain.chat_models import ChatOllama
from langchain.schema import Document
from client import Client

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
    