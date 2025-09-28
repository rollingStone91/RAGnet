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
import yaml
import requests
import logging
from datetime import datetime

# ====== 日志配置 ======
log_filename = f"./logs_algorithm/process_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
logging.basicConfig(
    level=logging.INFO,  # DEBUG 级别会记录更多细节
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(log_filename, encoding="utf-8"),  # 保存到文件
        logging.StreamHandler()  # 同时输出到控制台
    ]
)

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

API_URL = "https://dashscope.aliyuncs.com/api/v1/services/aigc/text-generation/generation"
API_KEY = "sk-3eef22f7794b4d2aaefbdd719a285208"

class Server_with_Algorithm:
    """
    Server 类，负责：
    1) 并行化多客户端检索
    2) 验证数据完整性（通过 Proof 信息）
    3) 调用 Ollama 部署的 Qwen3:4B 模型生成答案
    """
    def __init__(self, embedding=None, model_name: str = "qwen3:4b", base_url="http://4a7bdf20.r8.cpolar.cn"):
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
            
        self.proof_api = PrivacyProofAPI(base_url=base_url)  # Optional: PrivacyProofAPI 实例
        self.embeddings = embedding

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
            logging.info("原始向量维度:", index.d)
        self.cost.retrieval_time = time.time() - self.cost.start

        # 使用pedersen算法
        for p in proofs:
            # print(p.vector)
            # print(len(q_vec[0]))
            # print(f"Context: {p.document.page_content}")
            response = self.proof_api.gen_pedersen_proof(name="commonsense",
                                                            K=p.vector, Q=q_vec[0], 
                                                            data=p.document.page_content)
            logging.info(f"生成的pedersen proof:{response}")

            # response = self.proof_api.gen_groth_proof(name="commonsense",
            #                                                 K=p.vector, Q=q_vec[0], 
            #                                                 data=p.document.page_content)
            # logging.info(f"生成的groth proof:{response}")
            # p.groth_id = response["proof_id"]

            p.pedersen_id = response["proof_id"]
            self.cost.por_proof_size += response["space_cost"]
            self.cost.por_proof_time += response["time_cost"]

        return proofs, q_vec[0]

    def build_prompt(self, background:str, query: str, contexts: List[str], metadatas) -> str:
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
    
    def pog_verify(self, q_vec, proofs):
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
            logging.info(f"生成的pog:{res}")

            # 验证proof
            msg = self.proof_api.verify_pog(p.pedersen_id, p.pog_id)
            # msg = json.loads(msg)
            logging.info(f"验证pog的结果:{msg}")
            
            # 验证通过则加入列表
            if(msg['msg'] == "ok"):
                contexts.append(data)
                metadatas.append(p.document.metadata)
            self.cost.pog_verify_time += msg["time_cost"]

        proof_len = len(proofs) if len(proofs) > 0 else 1
        # 计算平均值
        self.cost.pog_proof_time = self.cost.pog_proof_time / proof_len
        logging.info(f"average pog time: {self.cost.pog_proof_time}")

        self.cost.pog_proof_size = self.cost.pog_proof_size / proof_len
        logging.info(f"average pog size: {self.cost.pog_proof_size}")   

        self.cost.pog_verify_time = self.cost.pog_verify_time / proof_len
        logging.info(f"average pog verify time: {self.cost.pog_verify_time}")

        return contexts, metadatas
    
    def pederson_verify(self, proof_len, all_proofs):
        # 验证groth
        verified_proof = []
        # flatten proofs
        for p in all_proofs:
            response = self.proof_api.verify_pedersen_proof(proof_id=p.pedersen_id)
            # response = json.loads(response)
            logging.info(f"verify pedersen proof:{response}")

            if(response['msg'] == "ok"):
                verified_proof.append(p)
            self.cost.por_verify_time += response["time_cost"]

        self.cost.por_verify_time = self.cost.por_verify_time / proof_len
        logging.info(f"average por verify_time:{self.cost.por_verify_time}")
        return verified_proof
    
    def groth_verify(self, proof_len, all_proofs):
        # 验证groth
        verified_proof = []
        # flatten proofs
        for p in all_proofs:
            response = self.proof_api.verify_groth_proof(proof_id=p.groth_id)
            # response = json.loads(response)
            logging.info(f"verify groth proof:{response}")

            if(response['msg'] == "ok"):
                verified_proof.append(p)
            self.cost.por_verify_time += response["time_cost"]
            
        self.cost.por_verify_time = self.cost.por_verify_time / proof_len
        logging.info(f"average por verify_time:{self.cost.por_verify_time}")
        return verified_proof

    def generate_answer(self, background:str, query: str, contexts: List[str], metadatas) -> str:
        prompt = self.build_prompt(background, query, contexts, metadatas)

        self.cost.start = time.time()
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
            self.cost.generation_time = time.time() - self.cost.start
            logging.info(f"LLM生成时间: {self.cost.generation_time}")

            answer = self.clean_answer(raw)
            return answer
        else:
            response = self.llm.invoke(prompt)
            self.cost.generation_time = time.time() - self.cost.start
            logging.info(f"LLM生成时间: {self.cost.generation_time}")

            answer = self.clean_answer(response.content)
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
        logging.info(f"average por proof size: {self.cost.por_proof_size}")

        self.cost.por_proof_time = self.cost.por_proof_time / proof_len
        logging.info(f"average por proof time: {self.cost.por_proof_time}")

        verified_proof = self.pederson_verify(proof_len, all_proofs)

        # 根据得分进行排序，选出最优proofs
        verified_proof.sort(key=lambda p: getattr(p, 'score', 0), reverse=True)

        # 取 Top-K
        selected = verified_proof[:top_k]

        # 验证pog
        contexts, metadatas = self.pog_verify(q_vec, selected)
        
        #生成答案并清洗
        answer = self.generate_answer(background=background, query=query, conetexts=contexts, metadatas=metadatas)

        return self.cost, answer
    