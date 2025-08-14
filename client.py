import os
from langchain_community.document_loaders import PyPDFLoader
from typing import List, Tuple, Dict, Union
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.tools.retriever import create_retriever_tool
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_community.embeddings import DashScopeEmbeddings
from datasets import load_dataset
from langchain.schema import Document
from langchain_experimental.text_splitter import SemanticChunker
import numpy as np
import json
# from llama_cpp import Llama
import faiss
import torch
from langchain.embeddings.base import Embeddings
import pandas as pd
import glob
import gzip
from langchain_community.embeddings import HuggingFaceEmbeddings
from sentence_transformers import SentenceTransformer
import re

# from modelscope import LLM, PoolingParams

class Proof():
    def __init__(self, document: Document, vector: List[np.ndarray]=[], score: float=0):
        self.document = document
        self.vector = vector
        self.score = score
        self.pedersen_id = 0
        self.groth_id = 0
        self.pog_id = 0

class QwenEmbeddings(Embeddings):
    """
    自定义维度的Embedding包装类，支持截取前N维
    """
    def __init__(self, model_name="./models/qwen3-embedding-0.6b", device="cuda", batch_size=8):
        self.device = device
        self.model = SentenceTransformer(
                                model_name,
                                model_kwargs={
                                    # "torch_dtype": torch.float16, 
                                    # "attn_implementation": "flash_attention_2", 
                                    "device_map": self.device},
                                # tokenizer_kwargs={"padding_side": "left"},
                                trust_remote_code=True)
        self.batch_size = batch_size

    def embed_documents(self, texts):
        # batch encode 文本列表，返回二维numpy数组
        embeddings = self.model.encode(texts, batch_size=self.batch_size, convert_to_numpy=True, normalize_embeddings=True)
        return embeddings.tolist()

    def embed_query(self, text):
        # encode 单个文本，返回list形式
        embedding = self.model.encode(text, convert_to_numpy=True, normalize_embeddings=True, prompt_name="query")
        return embedding.tolist()

    # def embed_documents(self, texts: List[str]) -> List[List[float]]:
    #     embeddings = self.embeddings.embed_documents(texts)
    #     if self.dim is not None:
    #         embeddings = [emb[:self.dim] for emb in embeddings]
    #     return embeddings
    
    # def embed_query(self, text: str) -> List[float]:
    #     embedding = self.embeddings.embed_query(text)
    #     if self.dim is not None:
    #         embedding = embedding[:self.dim]
    #     return embedding


class Client:
    """
    轻量级rag客户端，负责数据集加载、向量存储构建与检索。
    """
    def __init__(self, embedding, vectorstore_path: str = "faiss_db", min_len = 50): # dashscope_api_key: str,使用api调用embedding模型
        os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
        self.vectorstore_path = vectorstore_path
        self.embeddings = embedding
        self.db: FAISS = None
        self.min_len = min_len  # 低于这个字符数的块，认为过短

        # embeddings = HuggingFaceEmbeddings( model_name=model_path,
        #                                     model_kwargs={"device": "cuda"},
        #                                     encode_kwargs={"normalize_embeddings": True,
        #                                                     "batch_size": 8,})
        #                                     # multi_process=True)

        # self.embeddings = DashScopeEmbeddings(
        #     model="text-embedding-v1",
        #     dashscope_api_key=dashscope_api_key
        # )
        # self.embeddings = LlamaCppEmbeddings(model_path=model_path)
    
    # 读取PDF文件并提取文本内容
    def _read_pdfs(self, pdf_paths: List[str]) -> List[Document]:
        docs = []
        for i, path in enumerate(pdf_paths):
            loader = PyPDFLoader(path)
            pages = loader.load_and_split()
            for page in pages:
                docs.append(Document(page_content=page.page_content, metadata={'source': path, 'doc_id': i}))
        return docs

    # 读取JSON文件夹中的所有文件
    def _load_json_folder(self, folder_path: str, start=0, step=1000) -> List[Document]:
        docs = []
        json_files = sorted([f for f in os.listdir(folder_path) if f.endswith('.json')])
        selected_files = json_files[start:start + step]  # 选取指定范围的文件
        for i, filename in enumerate(selected_files):
            filepath = os.path.join(folder_path, filename)
            with open(filepath, encoding='utf-8') as f:
                data = json.load(f)
            content = f"{data.get('title', '')}\n{data.get('content', '')}".strip()
            if content:
                docs.append(Document(page_content=content, metadata={'source': filepath, 'doc_id': i + start}))
        return docs
    
    # 在线读取数据集
    def _streaming_load_dataset(self, sample_size=100, language='en', date_version='20231101') -> List[str]:
        # 启用streaming模式在线读取huggingface datasets
        dataset = load_dataset("wikimedia/wikipedia", f'{date_version}.{language}', streaming=True)
        docs = []
        for i, item in enumerate(dataset['train']):
            if i >= sample_size:
                break
            text = item.get('text', '').strip()
            title = item.get('title', '').strip()
            if not text:
                continue
            # snippet = text[:5000]
            meta = {'source': f'wikipedia://{language}/{item.get("id")}', 'doc_id': i}
            docs.append(Document(page_content=f"{title}\n{text}", metadata=meta))
        print(f"Streamed {len(docs)} Wikipedia docs.")
        return docs
    
    def _load_pubmedqa(self, data_files: Union[str, List[str]]="datasets/pubmedqa/pqa_artificial/train-00000-of-00001.parquet") -> List[Document]:
        """
        从本地 parquet 文件加载 PubMedQA 数据集，输出 Document 列表。
        data_files: 单个文件路径或路径列表。
        """
        pubmedqa_ds = load_dataset("parquet", data_files=data_files, split="train")
        docs = []
        for ex in pubmedqa_ds:
            pubid = ex.get('pubid', '')
            question = ex.get('question', '')
            answer = ex.get('final_decision','')
            long_answer = ex.get('long_answer', '')
            contexts = ex["context"]["contexts"]
            labels = ex["context"]["labels"]
            meshes = ex["context"].get("meshes", [])
            context_text = f"Question:{question}"
            context_text += "\n".join([f"{label}:{text}" for label, text in zip(labels, contexts)])
            if context_text.strip():
                docs.append(Document(page_content=context_text,
                                     metadata={'source': data_files, 'doc_id': pubid, 
                                               'answer': answer, 'long_answer': long_answer,
                                                'meshes': meshes}))
        return docs
    
    def _load_legalbench(self, data_dir: str ="./datasets/legalbench/data", tasks: Union[str, List[str]]="abercrombie") -> List[Document]:
        """
        从本地下载的 LegalBench 数据目录加载指定任务。
        支持 .tsv 格式，如 abercrombie/train.tsv；
        data_dir: 根目录；tasks: 单个或列表，任务名称。
        """
        docs = []
        tasks = [tasks] if isinstance(tasks, str) else tasks
        for task in tasks:
            task_dir = os.path.join(data_dir, task)
            tsv_path = os.path.join(task_dir, 'train.tsv')
            if not os.path.exists(tsv_path):
                continue
            df = pd.read_csv(tsv_path, sep='\t')
            for _, row in df.iterrows():
                text = str(row.get('text', '')).strip()
                answer = row.get('answer') or row.get('label', '')
                input_content = text
                metadata = {
                    'source': f'legalbench/{task}',
                    'task': task,
                    'idx': int(row.get('index', row.get('idx', 0))),
                    'answer': answer
                }
                docs.append(Document(page_content=input_content, metadata=metadata))
        return docs
    
    def _load_codesearchnet(self, path: Union[str, List[str]] = "./datasets/code_search_net/data/python/final/jsonl/train/*.jsonl.gz", 
                            language: str = 'python') -> List[Document]:
        """
        先把data目录下的每个zip文件解压
        然后从本地 CodeSearchNet .jsonl.gz 文件中加载 Document 列表
        path: 单个文件路径或路径通配符（如 'data/python/train/*.jsonl.gz'）
        language: 选择语言配置，如 'python', 'java', 'all'
        """
        file_list = glob.glob(path) if isinstance(path, str) else path
        docs = []
        for file in file_list:
            with gzip.open(file, 'rt', encoding='utf-8') as f:
                for i, line in enumerate(f):
                    try:
                        item = json.loads(line)
                        code = item.get("code") or item.get("original_string", "")
                        docstring = item.get("docstring", "")
                        lang = item.get("language") or language
                        text = f"[Language: {lang}]\n[Docstring]\n{docstring}\n[Code]\n{code}"
                        metadata = {
                        "repo": item.get("repo"),
                        "func": item.get("func_name"),
                        "path": item.get("path"),
                        "language": lang,
                        "source": file,
                        "url": item.get("url")
                        }
                        docs.append(Document(page_content=text, metadata=metadata))
                    except Exception as e:
                        print(f"Error parsing line {i} in {file}: {e}")
        return docs

    def _merge_short_chunks(self, chunks: List[str]) -> List[str]:
        """
        更稳健的短块合并逻辑：
        - 将长度 < min_len 的块优先合并到前一个块；若不存在前一个块，则合并到下一个块。
        - 合并完成后剔除依然过短（<=20）或只含标点/空白的块。
        - 保证不会无限增长（如需可在外层再根据 max_chunk_size 切分）。
        """
        merged = []
        for chunk in chunks:
            chunk = chunk.strip()
            if not chunk:
                continue
            if len(chunk) < self.min_len:
                # 尝试合并到前一个
                if merged:
                    merged[-1] = (merged[-1] + " " + chunk).strip()
                else:
                    # 暂时保存到 merged 以便下一次循环可以合并
                    merged.append(chunk)
            else:
                merged.append(chunk)

        # 如果第一个仍然很短且后面存在，则把它合并到第二个
        if len(merged) >= 2 and len(merged[0]) < self.min_len:
            merged[1] = (merged[0] + " " + merged[1]).strip()
            merged.pop(0)

        # 最终过滤掉过短或无效块
        cleaned = [c for c in merged if len(c.strip()) > 20 and not re.fullmatch(r"\W+", c)]
        return cleaned

    def _chunk_text(self, text, chunk_size = 8000, overlap = 400,
                    semantic_max = 2000,
                    buffer_size = 3, breakpoint_type = "percentile", sentence_split_regex = r"(?<=[.?!])\s+"):
        """
        两阶段切分（coarse->semantic）策略：
        1) 先用较大的递归字符切分器把文档切为较大的窗（coarse_chunk_size），以减少对SemanticChunker的调用次数（节省计算和显存）。
        2) 对于较长的coarse chunk，再用SemanticChunker做语义切分，得到更精细、语义自洽的块。
        3) 合并过短块，并返回字符串列表。
        """
        # 预清理：合并多余空白
        text = re.sub(r"\s+", " ", text).strip()

        # 粗切分（大窗）
        coarse_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=overlap,
            separators=["\n\n", "\n", " ", "", "."],
            length_function=len,
        )
        coarse_chunks = coarse_splitter.split_text(text)

        out_chunks = []

        # 对每个coarse chunk决定是否需要semantic切分
        for c in coarse_chunks:
            c = c.strip()
            if not c:
                continue
            # 若长度小于semantic_max，直接作为候选（避免不必要的语义切分）
            if len(c) <= semantic_max:
                out_chunks.append(c)
                continue

            # 否则调用 SemanticChunker 进行语义级切分（减少语义切分次数）
            splitter = SemanticChunker(
                embeddings=self.embeddings,
                buffer_size=buffer_size,
                breakpoint_threshold_type=breakpoint_type,
                sentence_split_regex=sentence_split_regex,
            )
            docs = [Document(page_content=c)]
            sem_chunks = splitter.split_documents(docs)

            # 提取文本并合并短块
            texts = [d.page_content.strip() for d in sem_chunks if d.page_content.strip()]
            merged = self._merge_short_chunks(texts)
            out_chunks.extend(merged)

        # 最终清洗：去重、过滤、并返回
        # 使用简单去重保持顺序
        seen = set()
        final_chunks = []
        for chunk in out_chunks:
            key = chunk[:200]  # 截取开头作为快速去重的hash
            if key in seen:
                continue
            seen.add(key)
            final_chunks.append(chunk)
        return final_chunks
    
    def iter_doc_chunks(self, doc: Document, chunk_size = 8000, overlap: int = 400,
                        semantic_max = 2000, **semantic_kwargs):
        """
        将单篇文档流式化为Document块的生成器，便于分批嵌入与索引。
        """
        text = doc.page_content or ""
        chunks = self._chunk_text(text, chunk_size=chunk_size, overlap=overlap, semantic_max=semantic_max,
                                  **semantic_kwargs)
        for ch in chunks:
            yield Document(page_content=ch, metadata=doc.metadata)
    
    def build_vectorstore(self, docs:List[Document], batch_size=8, incremental=True):
        """
        构建向量数据库
        batch_size: 批处理大小
        incremental=True: 是否增量构建
        """
        # 支持增量构建：如已有索引，先加载
        if incremental and os.path.exists(self.vectorstore_path):
            self.load_vectorstore()

        texts_batch, metadatas_batch = [], []
        # 构建 FAISS
        for i, doc in enumerate(docs):
            for chunk_doc in self.iter_doc_chunks(doc):
                texts_batch.append(chunk_doc.page_content)
                metadatas_batch.append(chunk_doc.metadata)
                if len(texts_batch) >= batch_size:
                    if self.db is None:
                        self.db = FAISS.from_texts(
                                    texts_batch,
                                    embedding=self.embeddings,
                                    metadatas=metadatas_batch
                                )
                    else:
                        self.db.add_texts(texts_batch, metadatas=metadatas_batch)
                    
                    texts_batch.clear()
                    metadatas_batch.clear()
                    # 清理缓存，避免显存累积
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        torch.cuda.synchronize()
            print(f"Inserted batch up to docs {i+1}/{len(docs)}")

        # 保存向量库
        if self.db:
            self.db.save_local(self.vectorstore_path)
            print(f"Vectorstore saved to {self.vectorstore_path}")
        else:
            print("No data processed.")

    def load_vectorstore(self) -> None:
        """
        加载已保存的向量存储
        """
        if not os.path.exists(self.vectorstore_path):
            raise FileNotFoundError(f"Vectorstore directory '{self.vectorstore_path}' not found.")
        self.db = FAISS.load_local(
            self.vectorstore_path,
            embeddings=self.embeddings,
            allow_dangerous_deserialization=True
        )
        print(f"Vectorstore {self.vectorstore_path} loaded.")

    def retrieve(self, query:str, top_k=4, fetch_k: int = None):
        """
        通过query在FAISS向量库中检索k个最相似文档，
        返回每个Document对象、其特征向量及相似度得分
        """
        # 检查向量库是否已加载
        if self.db is None:
            raise ValueError("Vectorstore尚未加载，请先调用load_vectorstore或build_vectorstore")
        
        # 获取查询向量（HuggingFaceEmbeddings 已归一化输出）
        q_vec = self.embeddings.embed_query(query)
        query_vec = np.array(q_vec, dtype=np.float32)

        # 决定 fetch_k 大小，保证过滤后还能拿到 top_k
        if fetch_k is None:
            fetch_k = top_k * 5

        # 一次性用内积搜索 fetch_k 个候选（等价于余弦相似度）
        scores, indices = self.db.index.search(query_vec.reshape(1, -1), fetch_k)

        # 过滤掉过短文档，并截取前 top_k 个
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            doc_id = self.db.index_to_docstore_id[idx]
            doc = self.db.docstore.search(doc_id)
            if len(doc.page_content) < 20:
                continue
            vec = self.db.index.reconstruct(int(idx)).tolist()
            results.append(Proof(doc, vec, float(score)))
            if len(results) >= top_k:
                break

        return results, q_vec

if __name__ == "__main__":
    # 示例：创建Client并加载向量存储
    clients = [Client(vectorstore_path="./common_sense_db"), Client(vectorstore_path="./computer_science_coding_related_db"),
               Client(vectorstore_path="./law_related_db"), Client(vectorstore_path="./medicine_related_db")]
    folder_paths = ["./classified_dataset/common_sense", "./classified_dataset/computer_science_coding_related",
             "./classified_dataset/law_related", "./classified_dataset/medicine_related"]
    for c, f in zip(clients, folder_paths):
        print(f"Building vectorstore for {f}...")
        c.build_vectorstore(folder_path=f)
