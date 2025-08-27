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

# ====== 日志配置 ======
log_filename = f"./logs_4b_8b/process_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
logging.basicConfig(
    level=logging.INFO,  # DEBUG 级别会记录更多细节
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(log_filename, encoding="utf-8"),  # 保存到文件
        logging.StreamHandler()  # 同时输出到控制台
    ]
)

class Server:
    """
    Server 类，负责：
    1) 并行化多客户端检索
    2) 验证数据完整性（通过 Proof 信息）
    3) 调用 Ollama 部署的 Qwen3:4B 模型生成答案
    """
    def __init__(self, model_name: str = "qwen3:8b"):
        self.llm = ChatOllama(model=model_name)

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
        # 精炼指令：System + User 模式
        system_msg = """
        You are a professional Evidence-First QA Arbiter.
        Your mission is to answer factual questions with maximum precision for public knowledge verification tasks. **Think step by step.**
        You must follow all rules below regardless of domain.

        ---
        ## Step 0 – Domain Identification
        Before reasoning, classify the question into one of the following domains:
        - **General** – Wikipedia-like factual or historical.
        - **Academic** – Research paper style, e.g., arXiv, physics, CS, math.
        - **Legal** – Law, statutes, case law, LegalBench style.
        - **Biomedical** – Clinical or life sciences research, PubMed style.
        If uncertain, choose the domain whose rules will best ensure factual precision.

        ---
        ## Core Professional Principles (all domains)
        1. **Rewrite the question** in your own words to clarify the key entities, time ranges, geographic scope, and topic.
        2. **Construct a mental timeline or geographic scope** relevant to the question.
        3. **Individually evaluate each provided context** for relevance to the clarified question.
        4. If a context is irrelevant, **discard it completely** and do not use it in reasoning or the answer.
        5. Use statements from relevant contexts to support your answer — never guess or invent.
        6. If there is no supporting evidence, answer exactly: "I don’t know.",But you can use the knowledge that you already knew before.
        7. If there is conflicting evidence, prefer the most explicit, direct, recent, and specific source.
        8. Even if 20 or more contexts are given, you must still return a **direct answer to the question**, never a generic summary.
        9. The earlier a context appears, the higher its similarity to the question — but still verify relevance before use.
        
        ---
        ## Domain-specific reasoning rules:
        ### 1. General factual / Wikipedia-like
        - Prioritize chronological and geographic precision.
        - For historical events, validate dates, places, and named entities directly from context.

        ### 2. Academic / Research (e.g., arXiv)
        1. Identify the research field, methodology, and key findings.
        2. Preserve formal definitions, equations, and exact terminology.
        3. Distinguish between experimental results, theoretical predictions, and speculation.
        4. Prefer peer-reviewed, cited, and clearly supported statements over informal claims.

        ### 3. Legal / Case Law (e.g., LegalBench)
        1. Identify jurisdiction, applicable laws, and relevant time frame.
        2. Distinguish **binding precedent** from **persuasive authority**.
        3. Quote statutory or case law language exactly where relevant.
        4. If the question is about applicability, clearly state the controlling law and its source.
        5. If multiple jurisdictions are mentioned, clearly state which one applies.

        ### 4. Biomedical / Clinical Research (e.g., PubMed)
        1. Identify the clinical question type (mechanism, diagnosis, treatment, prognosis, etc.).
        2. Extract key PICO elements: Population, Intervention, Comparator, Outcome.
        3. Apply **evidence hierarchy**:  
           - Meta-analysis/Systematic Review > Randomized Controlled Trial > Cohort Study > Case-Control > Case Report.
        4. Report numerical results exactly (RR, OR, CI, p-values).
        5. Do not generalize from animal studies to humans unless explicitly stated in the context.
        6. For conflicting results, state the balance of evidence and level of certainty.

        ---
        ## Special reasoning requirement:
        - Before your final answer, show your reasoning enclosed in <think> ... </think>.
        - Your reasoning must:
          1. Rewrite the question.
          2. Determine time/space scope.
          3. Identify domain type.
          4. Apply relevant domain-specific rules.
          5. Filter contexts, listing which are used and why, and which are discarded.
          6. Construct the final answer based only on retained contexts.

        **Final Answer Restriction:**  
        The final answer must **not** include or reference any provided context text directly — only present the conclusion based on the retained evidence.
        Keep the answer as short and direct as possible — answer only what is asked, without adding extra information, explanations, or commentary.

        Formatting rules:
        - Show your reasoning inside <think> ... </think> before giving the final answer.
        - Always indicate in <think> which contexts were used and why, and which were discarded.
        - Always produce a single, direct answer, not a summary.
        """

        # 规则：
        # - Do NOT reveal or describe your reasoning process, chain-of-thought, analysis, or intermediate steps. Do not use <think> tags. Output only the final answer.
        # - You may use your own general/world knowledge in addition to the provided contexts and metadata. Prefer information from the provided material when it directly addresses the question.
        # - The earlier a context appears, the more likely it contains the correct answer — prioritize earlier contexts when reasoning.
        # - If the answer cannot be found in the contexts, reply exactly “I don't know.”
        # - Provide standardized answers based on the "descriptions" of the question. 

        # 关闭思考链
        # Do NOT reveal or describe your reasoning process, chain-of-thought, analysis, or intermediate steps. Do not use <think> tags. Output only the final answer.
        # Before the final answer, show your step-by-step reasoning prefixed with “<think>” and suffixed with “</think>”.
        
        # 允许使用外部知识
        # You may use your own general/world knowledge in addition to the provided contexts and metadata. Prefer information from the provided material when it directly addresses the question.
        # Use ONLY the following contexts and metadata to answer; do NOT use any outside knowledge.
        
        
        # Few‑Shot example
        general_few_shot = """
        ### General Example
        Question: "Did Francois Mitterrand ever meet Barack Obama while they both held the position of President?"
        Contexts:
        [Context 1] Sandro Pertini, Italian president... Franklin D. Roosevelt (1882–1945), 32nd President of the United States... Joseph Stalin (1878–1953), Premier of the USSR...
        [Context 2] In 2007, Downey conducted a forum... Senators Joe Biden, Hillary Clinton, Obama...
        [Context 3] April 23, 1964... Tanganyikan President Julius Nyerere...
        [Context 4] U.S. President Barack Obama commented on fake news in 2016...
        [Context 5] Milton R. Wolf... is President Barack Obama's second cousin...

        <think>     
        Step 1 - Rephrase: The question asks if Mitterrand and Obama were both presidents at the same time and met during that overlap.  
        Step 2 - Keywords: "Francois Mitterrand", "Barack Obama", "both presidents", "met".  
        Step 3 - Timeline check:
           - Mitterrand was President of France from 1981 to 1995.
           - Obama was President of the U.S. from 2009 to 2017.
           - Their terms did not overlap.
        Step 4 - Context check:
           - Context 1: Mentions many presidents, but not Mitterrand-Obama meeting. (Irrelevant)
           - Context 2: Obama appears in 2007 as a candidate, Mitterrand already out of office. (Irrelevant)
           - Context 3: Event in 1964, unrelated. (Irrelevant)
           - Context 4: Obama in 2016, Mitterrand long gone. (Irrelevant)
           - Context 5: Obama's cousin, no link to Mitterrand. (Irrelevant)
        Step 5 - Conclusion: No evidence of a meeting while both were presidents; historically impossible due to non-overlapping terms.
        </think>
        Answer: False
        """
        cus_few_shot = background["fewshot"]
        few_shot = cus_few_shot if cus_few_shot else general_few_shot
        # user_msg = f"Question: {query}/no_think\n"
        user_msg = f"Question: {query}\n"
        user_msg += f'Here is a description of the question: {background["Instruction"]}\n'
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
        # print(f"uncleaned answer: {raw}")
        logging.info(f"未处理过的模型生成答案: {raw}")
        # cleaned = raw
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
        selected = all_proofs[:top_k]
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
    