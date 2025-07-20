import os
from PyPDF2 import PdfReader
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
from llama_cpp import Llama
from langchain.embeddings.base import Embeddings

# DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY")

# 自定义 LangChain 的 Embeddings 类封装
class LlamaCppEmbeddings(Embeddings):
    def __init__(self, model_path: str):
        self.llm = Llama(model_path=model_path, embedding=True)

    def embed_documents(self, texts):
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
    
class Proof:
    """
    隐私证明数据结构
    """
    def __init__(self, doc_id: str, score: float, vector: List[float], proof_data: Any):
        self.doc_id = doc_id
        self.score = score
        self.vector = vector
        self.proof_data = proof_data

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

    def _chunk_text(self, text: str, chunk_size=1000, overlap= 200) -> list[str]:
        """
        将文本分块处理，使用递归字符分割器。
        """
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=overlap
        )
        return splitter.split_text(text)

    def _read_pdfs(self, pdf_paths: List[str]) -> str:
        text = []
        for path in pdf_paths:
            reader = PdfReader(path)
            for page in reader.pages:
                text.append(page.extract_text() or "")
        return "\n".join(text)

    def build_vectorstore_from_pdf(self, pdf_paths: list[str]) -> None:
        """
        处理pdf文件并构建FAISS向量存储。
        """
        raw = self._read_pdfs(pdf_paths)
        chunks = self._chunk_text(raw)
        self.db = FAISS.from_texts(chunks, embedding=self.embeddings)
        self.db.save_local(self.vectorstore_path)
        print(f"Vectorstore built at '{self.vectorstore_path}' with {len(chunks)} chunks.")

    def build_vectorstore_from_wiki(self, sample_size=100, batch_size=10):
        # 启用streaming模式在线读取huggingface datasets
        dataset = load_dataset("wikimedia/wikipedia", "20231101.en", streaming=True)
        iterator = iter(dataset["train"])
        texts = []
        count = 0
        for item in iterator:
            if count >= sample_size:
                break
            text = item.get("text", "")
            if text:
                texts.append(text)
                count += 1
        print(f"Total collected Wikipedia texts: {len(texts)}")

        # 分块并批量处理
        all_chunks = []
        for i, text in enumerate(texts):
            chunks = self._chunk_text(text)
            all_chunks.extend(chunks)

            # 每 batch_size 保存一次，防止内存溢出
            if len(all_chunks) >= batch_size or i == len(texts) - 1:
                if self.db is None:
                    self.db = FAISS.from_texts(all_chunks, embedding=self.embeddings)
                else:
                    self.db.add_texts(all_chunks)
                all_chunks.clear()
                print(f"Processed {i+1}/{len(texts)} articles...")

        # 保存向量库
        if self.db:
            self.db.save_local(self.vectorstore_path)
            print(f"Vectorstore saved to {self.vectorstore_path}")
        else:
            print("No data processed.")

    def load_vectorstore(self) -> None:
        """
        加载已保存的向量存储，并初始化检索器。
        """
        if not os.path.exists(self.vectorstore_path):
            raise FileNotFoundError(f"Vectorstore directory '{self.vectorstore_path}' not found.")
        self.db = FAISS.load_local(
            self.vectorstore_path,
            embeddings=self.embeddings,
            allow_dangerous_deserialization=True
        )
        self.retriever = self.db.as_retriever()
        print("Vectorstore loaded and retriever initialized.")

    def retrieve(self, query: str, top_k: int = 5) -> Tuple[List[Document], List[List[float]]]:
        """
        对输入 query 执行检索，返回 top_k 最相似文档及其向量。
        Output:
          docs: List[langchain.schema.Document]
          vectors: List[List[float]]
        """
        if self.db is None:
            raise RuntimeError("Vectorstore 未初始化，调用 load_vectorstore 或 build 方法先初始化。")
        # 生成 query 向量
        q_vec = self.embeddings.embed_query(query)
        print(q_vec)
        # 确保转换成 2D NumPy 数组，FAISS 要求 shape = (n_queries, dim)
        if isinstance(q_vec, list):
            q_vec = np.array(q_vec, dtype="float32").reshape(1, -1)
        # 使用 FAISS 原生 index.search
        D, I = self.db.index.search(q_vec, top_k)  # D: 距离, I: 索引ID
        docs = []
        vecs = []

        for idx in I[0]:
            if idx == -1:
                continue
            # 用 LangChain 的 docstore 获取 Document
            doc_id = self.db.index_to_docstore_id[idx]
            doc = self.db.docstore.search(doc_id)
            docs.append(doc)

            # 从 FAISS 中 reconstruct 向量
            vec = self.db.index.reconstruct(int(idx))
            vecs.append(vec)

        return docs, vecs
        # results = self.db.index.search(q_vec, top_k)
        # print(results)
        # ids, scores = results[1].tolist()[0], results[0].tolist()[0]
        # docs = [self.db.docstore.search(id) for id in ids]
        # vecs = [self.db.index.reconstruct(id) for id in ids]
        # return docs, vecs


    def query(self, question: str, top_k: int = 5) -> Tuple[List[Proof], List[float]]:
        """
        基于检索结果计算 Proof，并返回 proofs 列表及 query 向量。
        """
        docs, vecs = self.retrieve(question, top_k)
        proofs: List[Proof] = []
        for doc, vec in zip(docs, vecs):
            proof_data = {"doc_id": doc.metadata["source"], "merkle_path": []}
            proofs.append(Proof(doc.metadata.get("source", ""), score=0.0, vector=vec, proof_data=proof_data))
        q_vec = self.embeddings.embed_query(question)
        return proofs, q_vec

if __name__ == "__main__":
    # 初始化 RAG 客户端
    client = Client()
    # 构建或加载向量库
    # client.build_vectorstore_from_wiki(sample_size=100, batch_size=10)
    client.load_vectorstore()
    docs, vecs = client.retrieve("What is Abkhaz alphabet?", top_k=5)
    print(docs)
    print(vecs)