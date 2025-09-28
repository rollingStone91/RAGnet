from langchain.embeddings.base import Embeddings
from sentence_transformers import SentenceTransformer
import torch
import numpy as np
import torch
# import vllm    # Requires vllm>=0.8.5
# from vllm import LLM


class QwenEmbedding(Embeddings):
    """
    自定义维度的Embedding包装类，支持截取前N维
    """
    def __init__(self, model_name="./models/qwen3-embedding-0.6b", device="cuda", batch_size=4, output_dim=None):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model = SentenceTransformer(
                                model_name,
                                model_kwargs={
                                    "torch_dtype": torch.float16, 
                                    "attn_implementation": "flash_attention_2", 
                                    },
                                device=str(self.device),
                                tokenizer_kwargs={"padding_side": "left"},
                                trust_remote_code=True)
        self.batch_size = batch_size
        self.output_dim = output_dim
        
        # 关闭梯度，设置 eval，并把模型切换为半精度（如果在 CPU 上，半精度不会带来好处）
        self.model.eval()
        if self.device.type == "cuda":
            try:
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.set_float32_matmul_precision("high")
            except Exception:
                pass

        # 进一步确保不计算梯度
        for p in self.model.parameters():
            p.requires_grad = False
    
    def _truncate(self, embeddings):
        if self.output_dim is not None:
            return embeddings[:, :self.output_dim] if embeddings.ndim > 1 else embeddings[:self.output_dim]
        return embeddings
    

    def embed_documents(self, texts):
        """
        batch encode 文本列表
        """
        # SentenceTransformer 的encode已经做了batching，下面用inference_mode + autocast
        with torch.inference_mode():
            # autocast会在内部使用半精度运算以加速
            with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                embeddings = self.model.encode(
                    texts,
                    batch_size=self.batch_size,
                    convert_to_numpy=True,
                    normalize_embeddings=True,
                    show_progress_bar=False
                )
        embeddings = embeddings.astype(np.float32, copy=False)
        return self._truncate(embeddings)

    def embed_query(self, text):
        """encode 单个文本"""
        # SentenceTransformer 的encode已经做了batching，下面用inference_mode + autocast
        with torch.inference_mode():
            # autocast会在内部使用半精度运算以加速
            with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                embedding = self.model.encode(
                    text,
                    batch_size=self.batch_size,
                    convert_to_numpy=True,
                    normalize_embeddings=True,
                    show_progress_bar=False,
                    prompt_name="query"
                )
        embeddings = embeddings.astype(np.float32, copy=False)
        return self._truncate(embeddings)
    
    # class QwenEmbeddings(Embeddings):
    # """
    # 使用 vllm LLM 做文本 embedding 的包装类
    # """
    # def __init__(self, model_name="./models/qwen3-embedding-4b",
    #               batch_size=4,
    #             # Each query must come with a one-sentence instruction that describes the task
    #               task = 'Given a web search query, retrieve relevant passages that answer the query'):
    #     self.batch_size = batch_size

    #     # 初始化 vllm LLM
    #     self.model = LLM(
    #         model=model_name,
    #         task="embed",
    #         tensor_parallel_size=1,
    #         gpu_memory_utilization=0.3
    #     )

    #     self.instruction = task    
    
    # def get_detailed_instruct(self, query: str) -> str:
    #     return f'Instruct: {self.instruction}\nQuery:{query}'

    # def _normalize(self, arr: np.ndarray) -> np.ndarray:
    #     """L2 归一化"""
    #     norm = np.linalg.norm(arr, axis=-1, keepdims=True)
    #     return arr / np.clip(norm, 1e-12, None)
    
    # def embed_documents(self, texts):
    #     """
    #     批量 encode 文本列表
    #     """
    #     embeddings_list = []
    #     for i in range(0, len(texts), self.batch_size):
    #         batch_texts = texts[i:i+self.batch_size]
    #         outputs = self.model.embed(batch_texts)
    #         batch_embeddings = [o.outputs.embedding for o in outputs]
    #         embeddings_list.extend(batch_embeddings)

    #     return np.array(embeddings_list, dtype=np.float32)

    # def embed_query(self, text):
    #     """
    #     encode 单条文本
    #     """
    #     text = self.get_detailed_instruct(text)
    #     outputs = self.model.embed([text])
    #     embedding = outputs[0].outputs.embedding
    #     return np.array(embedding, dtype=np.float32)
