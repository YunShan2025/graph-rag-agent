from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain.callbacks.streaming_aiter import AsyncIteratorCallbackHandler
from langchain.callbacks.manager import AsyncCallbackManager

import os
from typing import List
import numpy as np

from graphrag_agent.config.settings import (
    TIKTOKEN_CACHE_DIR,
    OPENAI_EMBEDDING_CONFIG,
    OPENAI_LLM_CONFIG,
    CACHE_EMBEDDING_PROVIDER,
    CACHE_SENTENCE_TRANSFORMER_MODEL,
    MODEL_CACHE_DIR,
    EMBEDDING_BATCH_SIZE,
)

try:
    # optional local dependencies
    from sentence_transformers import SentenceTransformer
    import torch
except Exception:
    SentenceTransformer = None
    torch = None


# 设置 tiktoken 缓存目录，避免每次联网拉取
def setup_cache():
    TIKTOKEN_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    os.environ["TIKTOKEN_CACHE_DIR"] = str(TIKTOKEN_CACHE_DIR)


setup_cache()

def get_embeddings_model():
    """
    返回一个 embeddings 接口对象。

    如果环境变量或设置指向本地 sentence-transformers（`CACHE_EMBEDDING_PROVIDER='sentence_transformer'`），
    则返回一个轻量适配器，暴露 `embed_documents(texts)` 与 `embed_query(text)` 方法以兼容项目调用。
    否则回退到 OpenAIEmbeddings。
    """
    provider = (CACHE_EMBEDDING_PROVIDER or "").lower()

    if provider == "sentence_transformer" and SentenceTransformer is not None:
        class SentenceTransformerAdapter:
            def __init__(self, model_name: str = CACHE_SENTENCE_TRANSFORMER_MODEL):
                self.model_name = model_name
                # 选择设备：优先 GPU
                self.device = "cuda" if (torch is not None and torch.cuda.is_available()) else "cpu"
                # 将模型缓存到项目 model cache 目录，避免每次下载
                cache_folder = os.getenv("SENTENCE_TRANSFORMERS_CACHE") or str(MODEL_CACHE_DIR)
                try:
                    self.model = SentenceTransformer(self.model_name, cache_folder=cache_folder)
                except TypeError:
                    # older sentence-transformers may not support cache_folder kw
                    self.model = SentenceTransformer(self.model_name)
                try:
                    # 尝试把模型移动到 device（若支持）
                    if hasattr(self.model, 'to') and self.device == 'cuda':
                        self.model.to(self.device)
                except Exception:
                    pass

                # 获取 embedding 维度
                try:
                    self.embedding_size = int(self.model.get_sentence_embedding_dimension())
                except Exception:
                    self.embedding_size = None

            def embed_documents(self, texts: List[str]):
                # batch encode -> 返回 List[List[float]]
                batch_size = EMBEDDING_BATCH_SIZE or 32
                try:
                    arr = self.model.encode(
                        texts,
                        batch_size=batch_size,
                        show_progress_bar=False,
                        convert_to_numpy=True,
                        device=self.device,
                    )
                except TypeError:
                    # older versions may not accept device in encode
                    arr = self.model.encode(texts, batch_size=batch_size, show_progress_bar=False, convert_to_numpy=True)
                # 确保为 numpy 数组
                if not isinstance(arr, np.ndarray):
                    arr = np.array(arr, dtype=np.float32)

                # 更新 embedding_size
                try:
                    if self.embedding_size is None:
                        self.embedding_size = int(arr.shape[-1])
                except Exception:
                    pass

                # L2 归一化
                norms = np.linalg.norm(arr, axis=1, keepdims=True)
                norms[norms == 0] = 1.0
                arr = arr / norms

                return arr.astype(np.float32).tolist()

            def embed_query(self, text: str):
                try:
                    arr = self.model.encode(text, show_progress_bar=False, convert_to_numpy=True, device=self.device)
                except TypeError:
                    arr = self.model.encode(text, show_progress_bar=False, convert_to_numpy=True)
                # 确保为 numpy 数组
                if not isinstance(arr, np.ndarray):
                    arr = np.array(arr, dtype=np.float32)

                # 如果是二维（单个输入也可能返回二维），降到一维
                if arr.ndim == 2 and arr.shape[0] == 1:
                    arr = arr[0]

                # 更新 embedding_size
                try:
                    if self.embedding_size is None:
                        self.embedding_size = int(arr.shape[-1])
                except Exception:
                    pass

                # L2 归一化
                norm = np.linalg.norm(arr)
                if norm == 0:
                    norm = 1.0
                arr = arr / norm

                return arr.astype(np.float32).tolist()

        return SentenceTransformerAdapter()

    # 默认回退到 OpenAIEmbeddings
    config = {k: v for k, v in OPENAI_EMBEDDING_CONFIG.items() if v}
    return OpenAIEmbeddings(**config)


def get_llm_model():
    config = {k: v for k, v in OPENAI_LLM_CONFIG.items() if v is not None and v != ""}
    return ChatOpenAI(**config)

def get_stream_llm_model():
    callback_handler = AsyncIteratorCallbackHandler()
    # 将回调handler放进AsyncCallbackManager中
    manager = AsyncCallbackManager(handlers=[callback_handler])

    config = {k: v for k, v in OPENAI_LLM_CONFIG.items() if v is not None and v != ""}
    config.update({"streaming": True, "callbacks": manager})
    return ChatOpenAI(**config)

def count_tokens(text):
    """简单通用的token计数"""
    if not text:
        return 0
    
    model_name = (OPENAI_LLM_CONFIG.get("model") or "").lower()
    
    # 如果是deepseek，使用transformers
    if 'deepseek' in model_name:
        try:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-V3")
            return len(tokenizer.encode(text))
        except:
            pass
    
    # 如果是gpt，使用tiktoken
    if 'gpt' in model_name:
        try:
            import tiktoken
            encoding = tiktoken.get_encoding("cl100k_base")
            return len(encoding.encode(text))
        except:
            pass
    
    # 备用方案：简单计算
    chinese = len([c for c in text if '\u4e00' <= c <= '\u9fff'])
    english = len(text) - chinese
    return chinese + english // 4

if __name__ == '__main__':
    # 测试llm
    llm = get_llm_model()
    print(llm.invoke("你好"))

    # 由于langchain版本问题，这个目前测试会报错
    # llm_stream = get_stream_llm_model()
    # print(llm_stream.invoke("你好"))

    # 测试embedding
    test_text = "你好，这是一个测试。"
    embeddings = get_embeddings_model()
    print(embeddings.embed_query(test_text))

    # 测试计数
    test_text = "Hello 你好世界"
    tokens = count_tokens(test_text)
    print(f"Token计数: '{test_text}' = {tokens} tokens")
