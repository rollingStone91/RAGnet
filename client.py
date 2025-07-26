import os
from langchain_community.document_loaders import PyPDFLoader
from typing import List, Tuple, Dict, Any
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

class Proof():
    def __init__(self, document: Document, vector: List[np.ndarray], score: float,
                  pedersen_id=None, groth_id=None):
        self.document = document
        self.vector = vector
        self.score = score
        self.pedersen_id = pedersen_id
        self.groth_id = groth_id



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
                vectorstore_path: str = "faiss_db"): # dashscope_api_key: str,使用api调用embedding模型
        os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
        self.vectorstore_path = vectorstore_path
        # self.embeddings = DashScopeEmbeddings(
        #     model="text-embedding-v1",
        #     dashscope_api_key=dashscope_api_key
        # )
        self.embeddings = HuggingFaceEmbeddings(
                                                model_name=model_path
                                                )
        # self.embeddings = LlamaCppEmbeddings(model_path=model_path)
        # model_kwargs={"device": "cpu"}
        self.db: FAISS = None

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
        return splitter.split_text(text)
    
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
        return splitter.split_documents(docs)

    # 读取PDF文件并提取文本内容
    def _read_pdfs(self, pdf_paths: List[str]) -> List[Document]:
        docs = []
        for i, path in enumerate(pages):
            loader = PyPDFLoader(path)
            pages = loader.load_and_split()
            for page in pages:
                docs.append(Document(page_content=page.page_content, metadata={'source': path, 'doc_id': i}))
        return docs

    # 读取JSON文件夹中的所有文件
    def _load_json_folder(self, folder_path: str) -> List[Document]:
        docs = []
        i = 0
        for filename in os.listdir(folder_path):
            if not filename.endswith('.json'):
                continue
            filepath = os.path.join(folder_path, filename)
            with open(filepath, encoding='utf-8') as f:
                data = json.load(f)
            content = f"{data.get('title', '')}\n{data.get('content', '')}".strip()
            if content:
                docs.append(Document(page_content=content, metadata={'source': filepath, 'doc_id': i}))
                i+=1
        return docs
    
    # 在线读取数据集
    def _streaming_load_dataset(self, sample_size=100, language='en', date_version='20231101') -> List[str]:
        # 启用streaming模式在线读取huggingface datasets
        dataset = load_dataset("wikimedia/wikipedia", f'{date_version}.{language}', streaming=True)
        docs = []
        for i, item in enumerate(dataset['train']):
            if i >= sample_size:
                break
            text = item.get('text', '')
            title = item.get('title', '')
            if not text:
                continue
            # snippet = text[:5000]
            meta = {'source': f'wikipedia://{language}/{item.get("id")}', 'doc_id': i}
            docs.append(Document(page_content=f"{title}\n{text}", metadata=meta))
        print(f"Streamed {len(docs)} Wikipedia docs.")
        return docs
    
    def build_vectorstore(self, sample_size=100, batch_size=10, 
                          streaming=False, folder_path=None, pdf_paths:List[str]=None,
                          buffer_size=3, threshold_type="percentile", sentence_split_regex=r"(?<=[.?!])\s+"):
        docs = []
        if streaming:
            # 在线读取数据集
            docs.extend(self._streaming_load_dataset(sample_size))
        elif folder_path is not None and pdf_paths is None:
            # 从指定文件夹加载JSON文件
            docs.extend(self._load_json_folder(folder_path))
        elif pdf_paths is not None:
            # 从PDF文件加载
            docs.extend(self._read_pdfs(pdf_paths))

        # 构建 FAISS
        texts, metadatas = [], []
        faiss_id = 0
        for doc in docs:
            # 为了避免显存爆炸，首先对文档进行字符切块
            chunks = self._chunk_text(doc.page_content)
            print(f"Total chunks after Character split: {len(chunks)}")
            for c in chunks:
                # 再使用SemanticChunker分块
                semantic_chunk = self._semantic_chunk_docs([Document(page_content=c, metadata=doc.metadata)], 
                                                           buffer_size, threshold_type, sentence_split_regex)
                print(f"Total chunks after semantic split: {len(semantic_chunk)}")

                # 把短块和前一个块合并
                MIN_LEN = 50  # 低于这个字符数的块，认为过短
                merged = []
                for chunk_doc in semantic_chunk:
                    text = chunk_doc.page_content
                    if len(text.strip()) < MIN_LEN and merged:
                        # 短块，且已有前一个块，则合并
                        prev = merged[-1]
                        # 保持前一个 Document 的 metadata 不变，仅拼接文本
                        prev.page_content = prev.page_content + " " + text
                    else:
                        # 正常块，直接添加
                        merged.append(Document(page_content=text, metadata=chunk_doc.metadata))

                # 用merged进行后续处理
                for i, d in enumerate(merged):
                    faiss_id += 1
                    texts.append(d.page_content)
                    metadatas.append({
                        "source": d.metadata.get("source", ""),
                        "doc_id": d.metadata.get("doc_id", ""),
                        "faiss_id": faiss_id
                    })
                    if len(texts) >= batch_size or i == len(merged) - 1:
                        if self.db is None:
                            self.db = FAISS.from_texts(
                                texts,
                                embedding=self.embeddings,
                                metadatas=metadatas,
                                normalize_L2=True
                            )
                        else:
                            self.db.add_texts(texts, metadatas=metadatas)
                        texts.clear()
                        metadatas.clear()
                        print(f"Inserted batch up to merged chunk {i+1}/{len(merged)}")
            # 清理缓存，避免显存累积
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()

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
