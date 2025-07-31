import os
from langchain_community.document_loaders import PyPDFLoader
from typing import List, Tuple, Dict, Union
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS
from langchain.tools.retriever import create_retriever_tool
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_community.embeddings import DashScopeEmbeddings
from datasets import load_dataset
from langchain.schema import Document
from langchain_experimental.text_splitter import SemanticChunker
from langchain.embeddings import HuggingFaceEmbeddings
import numpy as np
import json
# from llama_cpp import Llama
import faiss
import torch
from langchain.embeddings.base import Embeddings
import pandas as pd
import glob
import gzip

class Proof():
    def __init__(self, document: Document, vector: List[np.ndarray]=[], score: float=0):
        self.document = document
        self.vector = vector
        self.score = score
        self.pedersen_id
        self.groth_id
        self.pog_id


# # 自定义 LangChain 的 Embeddings 类封装
# class LlamaCppEmbeddings(Embeddings):
#     def __init__(self, model_path: str):
#         self.llm = Llama(model_path=model_path, embedding=True)

#     def embed_documents(self, texts: list[str]):
#         # return [self.llm.embed(text)["data"][0]["embedding"] for text in texts]
#         embeddings = []
#         for text in texts:
#             result = self.llm.embed(text)
#             if isinstance(result, list) and isinstance(result[0], list):
#                 embeddings.append(result[0])
#             else:
#                 embeddings.append(result)
#         return embeddings

#     def embed_query(self, text):
#         # return self.llm.embed(text)["data"][0]["embedding"]
#         result = self.llm.embed(text)
#         return result[0] if isinstance(result, list) and isinstance(result[0], list) else result
    
class Client:
    """
    轻量级rag客户端，负责数据集加载、向量存储构建与检索。
    """
    def __init__(self, model_path: str = "./models/qwen3-embedding-0.6b", 
                vectorstore_path: str = "faiss_db", MIN_LEN = 50): # dashscope_api_key: str,使用api调用embedding模型
        os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
        self.vectorstore_path = vectorstore_path
        self.embeddings = HuggingFaceEmbeddings(model_name=model_path,
                                                model_kwargs={"device": "cuda"},
                                                encode_kwargs={"normalize_embeddings": True},)
                                                # multi_process=True)
        self.db: FAISS = None
        self.MIN_LEN = MIN_LEN  # 低于这个字符数的块，认为过短
        # self.embeddings = DashScopeEmbeddings(
        #     model="text-embedding-v1",
        #     dashscope_api_key=dashscope_api_key
        # )
        # self.embeddings = LlamaCppEmbeddings(model_path=model_path)

    def _merge_short_chunks(self, chunks: List[str]) -> List[str]:
        """
        合并过短块（< MIN_LEN），避免出现无意义内容。
        """
        merged_chunks = []
        for c in chunks:
            if len(c) < self.MIN_LEN:
                if merged_chunks:
                    merged_chunks[-1] += " " + c  # 合并到前一个块
                else:
                    merged_chunks.append(c)  # 第一个块直接保留
            else:
                merged_chunks.append(c)
        # 再进行一次筛选，直接舍去合并后长度仍旧太小的块
        merged_chunks = [c.strip() for c in merged_chunks if len(c.strip()) > 20]
        return merged_chunks

    def _chunk_text(self, text: str, chunk_size=4000, overlap= 200) -> list[str]:
        """
        将文本分块处理，使用递归字符分割器。
        """
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=overlap,
            separators=["\n\n", "\n", " ", ""],
            length_function=len
        )
        chunks = splitter.split_text(text)
        # 过滤空块或纯符号块
        chunks = [c.strip() for c in chunks if len(c.strip()) > 0]
        return self._merge_short_chunks(chunks)
    
    def _semantic_chunk_docs(self, docs: list[Document],
                             buffer_size=3,
                             breakpoint_threshold_type="percentile",
                             sentence_split_regex=r"(?<=[.?!])\s+") -> list[Document]:
        """
        使用语义分块器对文档进行分块处理。
        """
        splitter = SemanticChunker(
            embeddings=self.embeddings,
            buffer_size=buffer_size,
            breakpoint_threshold_type=breakpoint_threshold_type,
            sentence_split_regex=sentence_split_regex
        )
        chunks = splitter.split_documents(docs)

        # 过滤空块并合并短块
        texts = [d.page_content.strip() for d in chunks if len(d.page_content.strip()) > 0]
        merged = self._merge_short_chunks(texts)
        return [Document(page_content=text) for text in merged]
    
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
    
    def build_vectorstore(self, batch_size=5, docs:List[Document]=[],                             
                        buffer_size=3, threshold_type="percentile",
                        sentence_split_regex=r"(?<=[.?!])\s+", incremental=True):
        """
        构建向量数据库
        batch_size:10批处理大小
        incremental=True: 是否增量构建
        """
        # 支持增量构建：如已有索引，先加载
        if incremental and os.path.exists(self.vectorstore_path):
            self.load_vectorstore()

        # 构建 FAISS
        texts, metadatas = [], []
        for j, doc in enumerate(docs):
            # try:    
            # 为了避免显存爆炸，首先对文档进行字符切块
            chunks = self._chunk_text(doc.page_content)
            print(f"Total chunks after Character split: {len(chunks)}")
            for c in chunks:
                # 再使用SemanticChunker分块
                semantic_chunk = self._semantic_chunk_docs([Document(page_content=c)], 
                                                            buffer_size=buffer_size, breakpoint_threshold_type=threshold_type,
                                                            sentence_split_regex=sentence_split_regex)
                print(f"Total chunks after semantic split: {len(semantic_chunk)}")
                # 用merged进行后续处理
                for i, d in enumerate(semantic_chunk):
                    texts.append(d.page_content)
                    metadatas.append({
                        #用来保存wiki数据集
                        "source": doc.metadata.get("source", ""),
                        "doc_id": doc.metadata.get("doc_id", ""),
                            #用来保存legalbench中的信息
                            # 'task': doc.metadata.get("task", ""),
                            # 'idx': doc.metadata.get("idx", ""),
                            # 'answer':doc.metadata.get("answer", ""),
                            #用来保存pubmedqa 
                            # 'long_answer': doc.metadata.get("long_answer",""),
                            # 'meshes': doc.metadata.get("meshes",""),
                            #用来保存codesearch
                            # "repo": doc.metadata.get("repository_name",""),
                            # "func": doc.metadata.get("func_name",""),
                            # "path": doc.metadata.get("func_path_in_repository",""),
                            # "language": doc.metadata.get("language",""),
                            # "url": doc.metadata.get("func_code_url","")
                    })
                    if len(texts) >= batch_size or i == len(semantic_chunk) - 1:
                        if self.db is None:
                            self.db = FAISS.from_texts(
                                texts,
                                embedding=self.embeddings,
                                metadatas=metadatas
                            )
                        else:
                            self.db.add_texts(texts, metadatas=metadatas)
                        texts.clear()
                        metadatas.clear()
                        print(f"Inserted batch up to merged chunk {i+1}/{len(semantic_chunk)}")
            # 清理缓存，避免显存累积
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            print(f"Inserted batch up to docs {j+1}/{len(docs)}")

            # except RuntimeError as e:
            #     if "out of memory" in str(e):
            #         # 保存已处理的部分到FAISS
            #         self.db.save_local(self.vectorstore_path)
            #         print(f"保存了{j}个doc文件到向量数据库中")
            #         if torch.cuda.is_available():
            #             torch.cuda.empty_cache()
            #             torch.cuda.synchronize()
            #         raise e  # 可选，或者直接终止程序
        
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

    def retrieve(self, query:str, top_k=4):
        """
        通过query在FAISS向量库中检索k个最相似文档，
        返回每个Document对象、其特征向量及相似度得分
        """
        # 检查向量库是否已加载
        if self.db is None:
            raise ValueError("Vectorstore尚未加载，请先调用load_vectorstore或build_vectorstore")
        
        # 查询向量并归一化
        query_vec = np.array(self.embeddings.embed_query(query), dtype=np.float32)
        query_norm = query_vec / np.linalg.norm(query_vec)

        # 使用 FAISS 内积搜索（等价于余弦相似度）
        scores, indices = self.db.index.search(query_norm.reshape(1, -1), top_k)

        results = []
        for i, idx in enumerate(indices[0]):
            if idx == -1:
                continue
            doc_id = self.db.index_to_docstore_id[idx]
            doc = self.db.docstore.search(doc_id)

            faiss_index = int(idx)
            vec = self.db.index.reconstruct(faiss_index).tolist()
            score = float(scores[0][i])  # 余弦相似度
            results.append(Proof(doc, vec, score))
        
        return results, query_norm.tolist()

if __name__ == "__main__":
    # 示例：创建Client并加载向量存储
    clients = [Client(vectorstore_path="./common_sense_db"), Client(vectorstore_path="./computer_science_coding_related_db"),
               Client(vectorstore_path="./law_related_db"), Client(vectorstore_path="./medicine_related_db")]
    folder_paths = ["./classified_dataset/common_sense", "./classified_dataset/computer_science_coding_related",
             "./classified_dataset/law_related", "./classified_dataset/medicine_related"]
    for c, f in zip(clients, folder_paths):
        print(f"Building vectorstore for {f}...")
        c.build_vectorstore(folder_path=f)
