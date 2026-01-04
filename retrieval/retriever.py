import json
import os
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional

# 尝试导入 rank_bm25
try:
    from rank_bm25 import BM25Okapi
    HAS_BM25 = True
except ImportError:
    HAS_BM25 = False

# 尝试导入 openai
try:
    from openai import OpenAI
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False


class Retriever:
    """Simple BM25-based retriever (Base Class)"""
    
    def __init__(self, corpus_path: str):
        self.corpus_path = Path(corpus_path)
        self.documents = []
        self.bm25 = None
        self._load_corpus()
        self._build_index()

    def _load_corpus(self):
        if not self.corpus_path.exists():
            raise FileNotFoundError(f"Corpus not found: {self.corpus_path}")
        self.documents = []
        with open(self.corpus_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    self.documents.append(json.loads(line))
        print(f"[Retriever] Loaded {len(self.documents)} documents from {self.corpus_path.name}")

    def _build_index(self):
        if not HAS_BM25:
            print("[WARNING] rank_bm25 not installed. Using simple matching.")
            return
        
        tokenized_docs = []
        for doc in self.documents:
            # 简单的空格分词，中文建议用 jieba
            text = f"{doc.get('title', '')} {doc.get('text', '')}"
            tokenized_docs.append(text.lower().split())
        self.bm25 = BM25Okapi(tokenized_docs)

    def search(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        # BM25 Logic (Keep your existing base logic here)
        if self.bm25:
            tokenized_query = query.lower().split()
            # 获取分数而不是直接由库排序，方便后续混合
            scores = self.bm25.get_scores(tokenized_query)
            top_n = np.argsort(scores)[::-1][:top_k]
            return [self.documents[i] for i in top_n]
        else:
            return self.documents[:top_k] # Fallback


class HybridRetriever(Retriever):
    """
    True Hybrid Retriever: BM25 + OpenAI Embeddings
    """

    def __init__(
        self,
        corpus_path: str,
        embedding_model: str = "text-embedding-3-small",
        bm25_weight: float = 0.5,
    ):
        super().__init__(corpus_path)
        self.embedding_model = embedding_model
        self.bm25_weight = bm25_weight
        self.doc_embeddings = None
        
        if not HAS_OPENAI:
            print("[ERROR] OpenAI not installed. Hybrid retriever implies usage of embeddings.")
            return
            
        # 注意：新版 SDK 参数名是 base_url
        self.client = OpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url=os.getenv("OPENAI_API_BASE")
        )
        
        # 加载或生成 Embeddings
        self._prepare_embeddings()

    def _get_embedding(self, text: str) -> List[float]:
        text = text.replace("\n", " ")
        return self.client.embeddings.create(input=[text], model=self.embedding_model).data[0].embedding

    def _prepare_embeddings(self):
        """Load embeddings from cache or generate them"""
        # 简单的缓存机制：同名文件加 .npy 后缀
        cache_path = self.corpus_path.with_suffix('.npy')
        
        if cache_path.exists():
            # print(f"[Retriever] Loading embeddings from cache: {cache_path}")
            self.doc_embeddings = np.load(cache_path)
            if len(self.doc_embeddings) != len(self.documents):
                print("[WARNING] Cache size mismatch. Regenerating...")
                self.doc_embeddings = None

        if self.doc_embeddings is None:
            print(f"[Retriever] Generating embeddings for {len(self.documents)} docs (this costs money)...")
            embeddings = []
            for doc in self.documents:
                # 组合标题和内容
                content = f"{doc.get('title', '')}: {doc.get('text', '')}"
                embeddings.append(self._get_embedding(content))
            
            self.doc_embeddings = np.array(embeddings)
            np.save(cache_path, self.doc_embeddings)
            print(f"[Retriever] Saved embeddings to cache.")

    def search(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """
        Weighted Hybrid Search
        Score = weight * BM25_Norm + (1-weight) * Vector_Norm
        """
        if self.doc_embeddings is None or self.bm25 is None:
            return super().search(query, top_k)

        # 1. 计算 BM25 分数
        tokenized_query = query.lower().split()
        bm25_scores = np.array(self.bm25.get_scores(tokenized_query))
        
        # 2. 计算 Vector 分数
        query_vec = np.array(self._get_embedding(query))
        # Cosine Similarity: (A . B) / (|A| * |B|)
        # 假设 OpenAI embedding 已经是归一化的，直接点积即可
        vector_scores = np.dot(self.doc_embeddings, query_vec)

        # 3. 归一化 (Min-Max Normalization) 防止量纲不同
        def normalize(scores):
            if np.max(scores) == np.min(scores):
                return np.zeros_like(scores)
            return (scores - np.min(scores)) / (np.max(scores) - np.min(scores))

        bm25_norm = normalize(bm25_scores)
        vector_norm = normalize(vector_scores)

        # 4. 加权融合
        final_scores = self.bm25_weight * bm25_norm + (1 - self.bm25_weight) * vector_norm

        # 5. 排序取 Top-K
        top_indices = np.argsort(final_scores)[::-1][:top_k]
        
        return [self.documents[i] for i in top_indices]