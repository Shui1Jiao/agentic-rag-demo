"""知识库检索层：语料分块 -> Chroma 建索引 -> 按语义相似度召回。

用 Chroma 自带的默认 embedding（本地 ONNX 版 all-MiniLM-L6-v2），
所以只跑检索不需要任何 API key，也不需要额外配置 embedding 服务。
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions

COLLECTION_NAME = "agentic_rag_demo"
# 索引落盘在仓库根目录下的 .chroma/，删掉即可重建
INDEX_DIR = Path(__file__).resolve().parent.parent / ".chroma"

# 语料按「一个知识点一条」切分。每条会被独立编码成向量，
# 检索时按语义相似度召回 Top-K，而不是把整段文本全塞进提示词。
DOCUMENTS: list[str] = [
    "Agentic-RAG：让 Agent 自主判断一个问题该走检索、调用工具还是直接回答，而不是固定执行「问题 → 检索 → 拼接 → 生成」的流水线。",
    "RAG（检索增强生成）：先从外部知识库检索相关内容，再把检索结果作为上下文交给大模型生成答案，用于缓解模型知识过时和幻觉。",
    "LangGraph：LangChain 官方推出的 Agent 编排框架，用状态图组织节点，核心概念是状态、条件分支和循环。",
    "StateGraph：LangGraph 的状态图容器，节点是普通的 Python 函数，接收状态并返回状态的增量更新。",
    "条件边（conditional edges）：LangGraph 中根据状态动态决定下一个节点的边，是 Agent 路由决策的实现手段。",
    "Function Calling：模型不直接回答，而是输出结构化的工具名与参数，由应用程序真正执行函数并把结果回灌给模型。",
    "向量数据库：存储文本的向量表示并支持相似度检索的数据库，常见的有 Chroma、FAISS、Milvus、pgvector。",
    "Chroma：轻量级本地向量数据库，支持内存与持久化两种模式，内置默认 embedding 模型，适合原型开发。",
    "FAISS：Facebook 开源的向量相似度检索库，只负责索引与检索，本身不含 embedding 模型，需要自己提供向量。",
    "Embedding：把文本映射成高维向量的过程，语义相近的文本在向量空间中距离更近，这是向量检索的基础。",
    "rerank（重排）：在向量召回之后，用更强的模型对候选片段重新打分排序，用于提升最终上下文的相关性。",
    "幻觉（hallucination）：大模型生成看似合理但事实上错误或无依据的内容，常见缓解手段是检索增强、引用来源与主动声明边界。",
]


@dataclass(frozen=True)
class Chunk:
    """一条检索结果。"""

    text: str
    source: str
    score: float


class KnowledgeBase:
    """基于 Chroma 的本地向量知识库，首次使用会自动建索引。"""

    def __init__(
        self,
        index_dir: Path | str = INDEX_DIR,
        embedding_function=None,
    ) -> None:
        self._client = chromadb.PersistentClient(
            path=str(index_dir),
            settings=Settings(anonymized_telemetry=False),
        )
        self._embedding_function = (
            embedding_function or embedding_functions.DefaultEmbeddingFunction()
        )
        self._collection = self._client.get_or_create_collection(
            name=COLLECTION_NAME,
            embedding_function=self._embedding_function,
            metadata={"hnsw:space": "cosine"},
        )
        self._build_if_needed()

    @property
    def size(self) -> int:
        """索引中的片段数量。"""
        return self._collection.count()

    def _build_if_needed(self) -> None:
        """语料与索引条数不一致时重建。upsert 是幂等的，重复执行安全。"""
        if self.size == len(DOCUMENTS):
            return
        self._collection.upsert(
            ids=[f"kb-{i:02d}" for i in range(len(DOCUMENTS))],
            documents=DOCUMENTS,
            metadatas=[{"source": f"kb#{i:02d}"} for i in range(len(DOCUMENTS))],
        )

    def search(self, query: str, top_k: int = 3) -> list[Chunk]:
        """按语义相似度召回 Top-K 片段；空查询或空库返回空列表。"""
        if not query.strip() or self.size == 0:
            return []

        result = self._collection.query(
            query_texts=[query],
            n_results=min(top_k, self.size),
        )
        documents = (result.get("documents") or [[]])[0]
        metadatas = (result.get("metadatas") or [[]])[0]
        distances = (result.get("distances") or [[]])[0]

        chunks: list[Chunk] = []
        for text, metadata, distance in zip(documents, metadatas, distances):
            # 集合使用 cosine 空间，distance = 1 - 余弦相似度
            score = 1.0 - float(distance)
            chunks.append(
                Chunk(
                    text=str(text),
                    source=str((metadata or {}).get("source", "?")),
                    score=max(0.0, min(1.0, score)),
                )
            )
        return chunks

    @staticmethod
    def format_context(chunks: list[Chunk]) -> str:
        """把召回片段拼成带来源标记的提示词上下文。"""
        return "\n".join(f"[{c.source}] {c.text}" for c in chunks)


@lru_cache(maxsize=1)
def get_knowledge_base() -> KnowledgeBase:
    """进程内单例，避免每次检索都重新加载索引与 embedding 模型。"""
    return KnowledgeBase()
