from typing import List, Tuple, Dict
from langchain.chat_models import ChatOllama
from langchain.schema import Document
from client import Client
import time
import re
import json
from langchain.schema import HumanMessage, SystemMessage
import logging
from datetime import datetime
import requests
import yaml

# ====== 日志配置 ======
log_filename = f"./logs_4b_instruct/process_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
logging.basicConfig(
    level=logging.INFO,  # DEBUG 级别会记录更多细节
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(log_filename, encoding="utf-8"),  # 保存到文件
        logging.StreamHandler()  # 同时输出到控制台
    ]
)

API_URL = "https://dashscope.aliyuncs.com/api/v1/services/aigc/text-generation/generation"
API_KEY = "sk-3eef22f7794b4d2aaefbdd719a285208"

class Server:
    """
    Server 类，负责：
    1) 并行化多客户端检索
    2) 验证数据完整性（通过 Proof 信息）
    3) 调用 Ollama 部署的 Qwen3:4B 模型生成答案
    """
    def __init__(self, model_name: str = "hopephoto/Qwen3-4B-Instruct-2507_q8:latest"):
        self.model_name = model_name
        self.api_url = API_URL
        self.api_key = API_KEY
        if model_name.startswith("qwen3-max") or model_name == "qwen3-max":
            self.llm = None
        else:
            self.llm = ChatOllama(model=model_name,
                                  reasoning=True,
                                  temperature=0.7,
                                  top_p=0.8,
                                  top_k=20,
                                  num_predict=16384)

    def _retrieve_from_clients(self, clients, query: str, top_k: int):
        proofs = []
        q_vec = []
        for client in clients:
            p, q = client.retrieve(query, top_k)
            proofs.extend(p)
            q_vec.append(q)
        return proofs, q_vec[0]

    def build_prompt(self, background, query: str, contexts: List[str], metadatas) -> str:
        """构造 Prompt，将 query 和上下文拼接"""

        with open("prompt.yaml", "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        system_msg = config["system_prompt"]["content"]

        # Few‑Shot example
        general_few_shot = config["fewshots"]["general"]["examples"]["content"]

        cus_few_shot = background["fewshot"]
        few_shot = cus_few_shot if cus_few_shot else general_few_shot

        # user_msg = f"Question: {query}/no_think\n"
        user_msg_template = config["user_prompts"]["template"]
        contexts_block = ""
        for i, (c, m) in enumerate(zip(contexts, metadatas)):
            contexts_block += f"[Context {i+1}] {c}\n[Metadata {i+1}] {json.dumps(m, ensure_ascii=False)}\n"
        
        user_msg = user_msg_template.format(query=query, contexts_block=contexts_block, instruction=background["Instruction"])
        human_msg = few_shot + "\n" + user_msg
        
        prompt = [SystemMessage(content=system_msg), HumanMessage(content=human_msg)]
        return prompt

    def clean_answer(self, raw: str):
        """
        去除 <think> 标签和其中的内容，并去掉多余空白
        """
        # 去掉所有 <think>…</think> 区段
        # print(f"uncleaned answer: {raw}")
        logging.info(f"未处理过的模型生成答案: {raw}")
        # cleaned = raw
        cleaned = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL)
        cleaned = re.sub(r"<tool_call>.*?</tool_call>", "", raw, flags=re.DOTALL)
        # print(f"cleaned answer: {cleaned}")

        # 从 LLM 回复中提取 'Final Answer' 后的内容
        match = re.search(r"Answer\s*:\s*<([^>]+)>", cleaned, flags=re.IGNORECASE) 
        if match:
            return match.group(1).strip()
        # 如果没有尖括号，就返回 Answer: 后的普通文本
        match2 = re.search(r"Answer\s*:\s*(.*)", cleaned, flags=re.IGNORECASE)
        if match2:
            return match2.group(1).strip()
        return cleaned.strip()
    
    def generate_answer(self, background:str, query: str, contexts: List[str], metadatas) -> str:
        prompt = self.build_prompt(background, query, contexts, metadatas)
        if self.model_name.startswith("qwen3-max") or self.model_name == "qwen3-max":
            message = [{
                'role':'system',
                'content': prompt[0].content
            }, {
                'role':'user',
                'content': prompt[1].content}]
            
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            payload = {
                "model": "qwen-max",
                "input": {
                    "messages": message,
                    "temperature": 0.7,
                    "top_p": 0.9
                }
            }
            try:
                resp = requests.post(self.api_url, headers=headers, json=payload, timeout=120)
                resp.raise_for_status()
                data = resp.json()
                raw = data.get("output", {}).get("text", "")
            except Exception as e:
                print(f"Qwen3-max API调用失败: {e}")
                raw = ""
            answer = self.clean_answer(raw)
            # print(f"answer:{answer}")
            return answer
        else:
            response = self.llm.invoke(prompt)
            answer = self.clean_answer(response.content)
            # print(f"answer:{answer}")
            return answer
    
    def multi_client_generate(self, background:dict, query:str, clients: List[Client], top_k=5):
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
        selected = all_proofs[:2]
        # 请求对应client提供真实上下文
        contexts = [r.document.page_content for r in selected]
        metadatas = [r.document.metadata for r in selected]

        
        for i, r in enumerate(selected):
            logging.info(f"选中的第{i+1}个上下文: {r.document.page_content}")

        # scores = [r.score for r in selected]
        # print(f"contexts: {contexts}") 
        # print(f"metadatas: {metadatas}")
        # print(f"scores: {scores}")
        
        #生成答案并清洗
        start = time.time()
        answer = self.generate_answer(background, query, contexts, metadatas)
        generate_latency = time.time() - start

        return retrieve_latency, generate_latency, contexts, answer
    