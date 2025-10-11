import os
from langchain_community.document_loaders import PyPDFLoader
from typing import List, Tuple, Dict, Union
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from datasets import load_dataset
from langchain.schema import Document
from langchain_experimental.text_splitter import SemanticChunker
import numpy as np
import json
import torch
import pandas as pd
import glob
import gzip
import re
from langchain_community.vectorstores.utils import (
    DistanceStrategy,
)

class Proof():
    def __init__(self, document: Document, vector: List[np.ndarray]=[], score: float=0):
        self.document = document
        self.vector = vector
        self.score = score
        self.pedersen_id = 0
        self.groth_id = 0
        self.pog_id = 0

class Client:
    """
    轻量级rag客户端，负责数据集加载、向量存储构建与检索。
    """
    def __init__(self, embedding, reranker=None, vectorstore_path: str = "faiss_db", min_len = 50): # dashscope_api_key: str,使用api调用embedding模型
        os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
        self.vectorstore_path = vectorstore_path
        self.embeddings = embedding
        self.reranker = reranker
        self.db: FAISS = None
        self.min_len = min_len  # 低于这个字符数的块，认为过短

        # 粗切分（大窗）
        self.coarse_splitter = RecursiveCharacterTextSplitter(
            chunk_size=8000,
            chunk_overlap=400,
            separators=["\n\n", "\n", " ", "", "."],
            length_function=len,
        )

        # 否则调用 SemanticChunker 进行语义级切分（减少语义切分次数）
        self.splitter = SemanticChunker(
            embeddings=self.embeddings,
            buffer_size=3,
            breakpoint_threshold_type="percentile",
            sentence_split_regex=r"(?<=[.?!])\s+",
        )
    
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
    def _load_json_folder(self, folder_path: str, start=0, end=1000) -> List[Document]:
        docs = []
        json_files = sorted([f for f in os.listdir(folder_path) if f.endswith('.json')])
        selected_files = json_files[start:end]  # 选取指定范围的文件
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
            context_text += f"\nLong Answer:{long_answer}\nSo the answer is {answer}."
            if context_text.strip():
                docs.append(Document(page_content=context_text,
                                     metadata={'pub_id': pubid, 'meshes': meshes}))
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

    def _chunk_text(self, text, semantic_max = 2000):
        """
        两阶段切分（coarse->semantic）策略：
        1) 先用较大的递归字符切分器把文档切为较大的窗（coarse_chunk_size），以减少对SemanticChunker的调用次数（节省计算和显存）。
        2) 对于较长的coarse chunk，再用SemanticChunker做语义切分，得到更精细、语义自洽的块。
        3) 合并过短块，并返回字符串列表。
        """
        # 预清理：合并多余空白
        text = re.sub(r"\s+", " ", text).strip()
        coarse_chunks = self.coarse_splitter.split_text(text)
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

            docs = [Document(page_content=c)]
            sem_chunks = self.splitter.split_documents(docs)

            # 提取文本并合并短块
            texts = [d.page_content.strip() for d in sem_chunks if d.page_content.strip()]
            merged = self._merge_short_chunks(texts)
            out_chunks.extend(merged)

        # 最终清洗：去重、过滤、并返回
        # 使用简单去重保持顺序
        seen = set()
        final_chunks = []
        for chunk in out_chunks:
            key = chunk[:256]  # 截取开头作为快速去重的hash
            if key in seen:
                continue
            seen.add(key)
            final_chunks.append(chunk)
        return final_chunks
    
    def iter_doc_chunks(self, doc: Document, semantic_max = 2000):
        """
        将单篇文档流式化为Document块的生成器，便于分批嵌入与索引。
        """
        text = doc.page_content or ""
        chunks = self._chunk_text(text, semantic_max=semantic_max)
        for ch in chunks:
            yield Document(page_content=ch, metadata=doc.metadata)

    def build_vectorstore(self, docs:List[Document], batch_size=4, incremental=True):
        """
        构建向量数据库
        batch_size: 批处理大小
        incremental=True: 是否增量构建
        """
        # 支持增量构建：如已有索引，先加载
        if incremental and os.path.exists(self.vectorstore_path):
            self.load_vectorstore()

        texts_batch, metadatas_batch = [], []
        for i, doc in enumerate(docs):
            for chunk_doc in self.iter_doc_chunks(doc):
                texts_batch.append(chunk_doc.page_content)
                metadatas_batch.append(chunk_doc.metadata)
                if len(texts_batch) >= batch_size:
                    if self.db is None:
                        self.db = FAISS.from_texts(
                                    texts_batch,
                                    embedding=self.embeddings,
                                    metadatas=metadatas_batch,
                                    **{"distance_strategy": DistanceStrategy.MAX_INNER_PRODUCT}
                                )
                    else:
                        self.db.add_texts(
                            texts_batch, 
                            metadatas=metadatas_batch,
                            )
                    
                    texts_batch.clear()
                    metadatas_batch.clear()
                    # 清理缓存，避免显存累积
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        torch.cuda.synchronize()
            print(f"Inserted batch up to docs {i+1}/{len(docs)}")

        if texts_batch:
            self.db.add_texts(
                texts_batch, 
                metadatas=metadatas_batch,
                )
            texts_batch.clear()
            metadatas_batch.clear()

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

    def retrieve(self, query:str, top_k=5, batch_size=5):
        """
        通过query在FAISS向量库中检索k个最相似文档，
        返回每个Document对象、其特征向量及相似度得分
        """
        # 检查向量库是否已加载
        if self.db is None:
            raise ValueError("Vectorstore尚未加载，请先调用load_vectorstore或build_vectorstore")
        
        # 获取查询向量（HuggingFaceEmbeddings 已归一化输出）
        q_vec = self.embeddings.embed_query(query)

        # 原生 FAISS 搜索，返回距离矩阵 D 和 索引矩阵 I
        D, I = self.db.index.search(q_vec.reshape(1, -1), top_k*2)  # 多取一些以便后续rerank

        contexts = []
        seen_texts = set()
        for dist, idx in zip(D[0], I[0]):
            if int(idx) < 0:
                continue

            # 将FAISS索引id映射到docstore id
            docstore_id = self.db.index_to_docstore_id[idx]

            # 从 docstore 取 Document；不同实现接口可能不同，这里使用内置 _dict 作为后备
            doc = self.db.docstore._dict[docstore_id]
            # 获取文本内容
            content = doc.page_content.strip()
            if not content or content in seen_texts:
                continue  # 跳过重复或空文本
            seen_texts.add(content)
            # reconstruct 向量（如果索引支持）    
            vec = self.db.index.reconstruct(int(idx)).tolist()

            # dist 的含义取决于索引类型，直接返回即可
            contexts.append(Proof(doc, vec, float(dist)))
        
        pairs = [(query, p.document.page_content) for p in contexts]
        all_scores = []

        # 禁用梯度，减少显存占用
        with torch.no_grad():
            for i in range(0, len(pairs), batch_size):
                batch_pairs = pairs[i:i+batch_size]
                inputs = self.reranker.process_inputs(batch_pairs)
                scores = self.reranker.compute_logits(inputs)
                all_scores.extend(scores)
                # 清理显存
                del inputs, scores
                torch.cuda.empty_cache()

        # 将分数赋值回proof对象替换原始的余弦值
        for c, s in zip(contexts, all_scores):
            c.score = s
        # 按分数排序取前 2
        top2 = sorted(contexts, key=lambda x: x.score, reverse=True)[:2]
        return top2, q_vec.tolist()
