import os
from langchain_community.document_loaders import PyPDFLoader
from typing import List, Tuple, Dict, Any
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS
from langchain.tools.retriever import create_retriever_tool
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_community.chat_models import ChatOllama
from langchain.chains import RetrievalQA
from datasets import load_dataset
from langchain.schema import Document
import numpy as np
import json
import faiss
from llama_cpp import Llama
from langchain.embeddings.base import Embeddings

class Proof():
    def __init__(self, document: Document, vector: List[np.ndarray], score: float):
        self.document = document
        self.vector = vector
        self.score = score

# 自定义 LangChain 的 Embeddings 类封装
class LlamaCppEmbeddings(Embeddings):
    def __init__(self, model_path: str):
        self.llm = Llama(model_path=model_path, embedding=True)

    def embed_documents(self, texts: list[str]):
        # return [self.llm.embed(text)["data"][0]["embedding"] for text in texts]
        embeddings = []
        for text in texts:
            result = self.llm.embed(text)
            if isinstance(result, list) and isinstance(result[0], list):
                embeddings.append(result[0])
            else:
                embeddings.append(result)
        return embeddings

    def embed_query(self, text):
        # return self.llm.embed(text)["data"][0]["embedding"]
        result = self.llm.embed(text)
        return result[0] if isinstance(result, list) and isinstance(result[0], list) else result
    
class Client:
    """
    轻量级rag客户端，负责数据集加载、向量存储构建与检索。
    """
    def __init__(self, model_path: str = "./models/Qwen3-Embedding/Qwen3-Embedding-0.6B-Q8_0.gguf", 
                vectorstore_path: str = "faiss_db"): # dashscope_api_key: str,使用api调用embedding模型
        os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
        self.vectorstore_path = vectorstore_path
        # self.embeddings = DashScopeEmbeddings(
        #     model="text-embedding-v1",
        #     dashscope_api_key=dashscope_api_key
        # )
        self.embeddings = LlamaCppEmbeddings(model_path=model_path)
        self.db: FAISS = None

    def _chunk_text(self, text: str, chunk_size=800, overlap= 200) -> list[str]:
        """
        将文本分块处理，使用递归字符分割器。
        """
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=overlap,
            separators=["\n\n", "\n", "。", ".", "！", "？", "!", "?", " ", ""],
            length_function=len
        )
        return splitter.split_text(text)

    # 读取PDF文件并提取文本内容
    def _read_pdfs(self, pdf_paths: List[str]) -> List[Document]:
        docs = []
        for path in pdf_paths:
            loader = PyPDFLoader(path)
            pages = loader.load_and_split()
            docs.extend(pages)
        return docs

    # 读取JSON文件夹中的所有文件
    def _load_json_folder(self, folder_path: str) -> List[Document]:
        docs = []
        for filename in os.listdir(folder_path):
            if not filename.endswith('.json'):
                continue
            filepath = os.path.join(folder_path, filename)
            with open(filepath, encoding='utf-8') as f:
                data = json.load(f)
            content = f"{data.get('title', '')}\n{data.get('content', '')}".strip()
            if content:
                docs.append(Document(page_content=content, metadata={'source': filepath}))
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
            meta = {'source': f'wikipedia://{language}/{item.get("id")}'}
            docs.append(Document(page_content=f"{title}\n{text}", metadata=meta))
        print(f"Streamed {len(docs)} Wikipedia docs.")
        return docs
    
    def build_vectorstore(self, sample_size=100, batch_size=10, 
                          streaming=False, folder_path=None, pdf_paths:List[str]=None):
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

        texts, metadatas = [], []
        faiss_id = 0
        # 分块并批量处理
        for i, doc in enumerate(docs):
            chunks = self._chunk_text(doc.page_content)
            for j, chunk in enumerate(chunks):
                texts.append(chunk)
                metadatas.append({
                "source": doc.metadata.get("source", ""),
                "doc_id": i,
                "chunk_id": j,
                "faiss_id": faiss_id
            })
                faiss_id += 1
                # 每 batch_size 保存一次，防止内存溢出
                if len(texts) >= batch_size or j == len(chunks) - 1:
                    if self.db is None:
                        self.db = FAISS.from_texts(texts, embedding=self.embeddings, metadatas=metadatas, normalize_L2=True)
                    else:
                        self.db.add_texts(texts, metadatas=metadatas)
                    texts.clear()
                    metadatas.clear()
            print(f"Processed {i+1}/{len(docs)} articles...")

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
        
        return results, query_vec
        # # 获取查询向量
        # query_vec = self.embeddings.embed_query(query)
        # query_vec = np.array(query_vec, dtype=np.float32)
        
        # # 使用FAISS索引进行搜索，获取L2距离和索引
        # distances, indices = self.db.index.search(np.array([query_vec], dtype=np.float32), top_k)

        # results = []
        # for i in range(top_k):
        #     index = indices[0][i]
        #     distance = distances[0][i]  # L2距离平方：||u - v||^2
        #     print(f"Index: {index}, Distance: {distance}")

        #     docstore_id = self.db.index_to_docstore_id[index]
        #     doc = self.db.docstore.search(docstore_id)

        #     faiss_index = int(index)
        #     vec = self.db.index.reconstruct(faiss_index)  # 重建文档向量
        #     vec = np.array(vec, dtype=np.float32)
        #     print(f"Document: {doc}, Vector: {vec}")

        #     # 计算向量范数
        #     query_norm = np.linalg.norm(query_vec)
        #     vec_norm = np.linalg.norm(vec)

        #     # 从L2距离计算内积：<u,v> = (||u||^2 + ||v||^2 - ||u - v||^2) / 2
        #     inner_product = (query_norm**2 + vec_norm**2 - distance) / 2

        #     # 计算余弦相似度
        #     cos_sim = inner_product / (query_norm * vec_norm) if query_norm * vec_norm != 0 else 0

        #     # 打包结果
        #     results.append(Proof(doc, vec.tolist(), cos_sim))
        # return results, query_vec

if __name__ == "__main__":
    # 示例：创建Client并加载向量存储
    client = Client(model_path="./models/Qwen3-Embedding/Qwen3-Embedding-0.6B-Q8_0.gguf", vectorstore_path="faiss_db")
    client.build_vectorstore(sample_size=100, batch_size=10, streaming=True)
    print("Client initialized and vectorstore built.")
    client.loud_vectorstore()
