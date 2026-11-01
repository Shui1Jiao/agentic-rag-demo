"""知识库检索层：文档切分 -> Chroma 建索引 -> 语义 + 字面混合召回。

语料来源优先 data/ 目录（.md / .txt / .docx / .pdf）；目录为空时回退到内置
DOCUMENTS，这样离线可跑、旧评测集仍与 BENCHMARK 可比。
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import chromadb
from chromadb.config import Settings

try:
    from src.documents import DATA_DIR, DocChunk, load_chunks
except ImportError:  # 兼容 python src/retriever.py 这类直接执行
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from src.documents import DATA_DIR, DocChunk, load_chunks


@lru_cache(maxsize=1)
def _default_embedding_function():
    """进程内复用的 ONNX embedding 实例。

    chromadb 的 DefaultEmbeddingFunction 每次 __call__ 都会新建 ONNXMiniLM_L6_V2()，
    而 ONNX 会话挂在实例的 cached_property 上 —— 等于每次检索都重新加载一遍模型。
    这里直接持有单例，实测单次编码 155.6ms -> 13.2ms。

    优先使用 项目内 models/ 目录下的模型（离线可用）；找不到再回退到
    chromadb 默认实现（会读用户缓存，必要时联网下载）。
    """
    from src.local_embedding import ProjectLocalONNX, model_is_present

    if model_is_present():
        return ProjectLocalONNX()

    from chromadb.utils.embedding_functions.onnx_mini_lm_l6_v2 import ONNXMiniLM_L6_V2

    return ONNXMiniLM_L6_V2()


def _env_flag(name: str, default: bool = True) -> bool:
    """读环境变量开关；未设置时用默认值。RAG_LOG=0 可静默。"""
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() not in {"0", "false", "no", "off"}


COLLECTION_NAME = "agentic_rag_demo"
# 语料/embedding 指纹清单（旁路文件，不碰 chroma 自己的 collection metadata）
MANIFEST_NAME = "kb_manifest.json"
# 索引落盘在仓库根目录下的 .chroma/，删掉即可重建
INDEX_DIR = Path(__file__).resolve().parent.parent / ".chroma"

# data/ 为空时的内置语料，与 BENCHMARK.md 的 12 条评测集保持一致
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


def builtin_chunks() -> list[DocChunk]:
    """内置语料 -> 片段列表（id 与 BENCHMARK 评测集对齐：kb-00..kb-11）。"""
    return [
        DocChunk(id=f"kb#{i:02d}", source=f"kb#{i:02d}", text=doc)
        for i, doc in enumerate(DOCUMENTS)
    ]


@lru_cache(maxsize=1)
def get_documents() -> tuple[DocChunk, ...]:
    """data/ 优先；目录里没有可用文件时回退到内置语料。"""
    loaded = load_chunks(DATA_DIR)
    return tuple(loaded) if loaded else tuple(builtin_chunks())


@dataclass(frozen=True)
class Chunk:
    """一条检索结果。source 形如 员工手册.md#03 或 kb#07。"""

    text: str
    source: str
    score: float


# —— 混合检索：向量召回 + 中文词面重排 ——
# 默认 embedding 是英文语料训练的 all-MiniLM-L6-v2，中文语义召回偏弱
# （20 条标注 query 上 recall@3 仅 65%）。这里在向量分之上叠加一层字符 bigram
# 覆盖率，不引入任何新模型或依赖，实测 recall@3 65% -> 85%。
RERANK_POOL = 6
LEXICAL_WEIGHT = 0.5


def _bigrams(text: str) -> set[str]:
    compact = "".join(ch for ch in text if not ch.isspace())
    return {compact[i : i + 2] for i in range(len(compact) - 1)}


def _lexical_score(query: str, document: str) -> float:
    """query 的字符 bigram 有多少比例出现在文档里（0~1）。"""
    query_bigrams = _bigrams(query)
    if not query_bigrams:
        return 0.0
    return len(query_bigrams & _bigrams(document)) / len(query_bigrams)


class KnowledgeBase:
    """基于 Chroma 的本地向量知识库，语料变了会自动重建索引。"""

    def __init__(
        self,
        index_dir: Path | str = INDEX_DIR,
        embedding_function=None,
        chunks: list[DocChunk] | None = None,
        verbose: bool | None = None,
    ) -> None:
        self._verbose = _env_flag("RAG_LOG", True) if verbose is None else verbose
        self._used_builtin = False
        stats: dict = {}
        self._index_dir = Path(index_dir)

        if chunks is not None:
            self._chunks = list(chunks)
        else:
            loaded = load_chunks(DATA_DIR, stats)
            self._chunks = list(loaded) if loaded else builtin_chunks()
            self._used_builtin = not loaded
        self._client = chromadb.PersistentClient(
            path=str(self._index_dir),
            settings=Settings(anonymized_telemetry=False),
        )
        self._embedding_function = (
            embedding_function or _default_embedding_function()
        )
        self._collection = self._client.get_or_create_collection(
            name=COLLECTION_NAME,
            embedding_function=self._embedding_function,
            metadata={"hnsw:space": "cosine"},
        )
        self._log_scan(stats)
        self._build_if_needed()

    # ---------- 启动反馈 ----------
    def _log(self, message: str) -> None:
        if self._verbose:
            print(f"[知识库] {message}")

    def _log_scan(self, stats: dict) -> None:
        if not self._verbose:
            return
        if stats:
            scanned = stats.get("scanned", 0)
            skipped = stats.get("skipped") or []
            detail = ""
            if skipped:
                items = "、".join(f"{s['name']}（{s['reason']}）" for s in skipped[:3])
                more = f" 等 {len(skipped)} 个" if len(skipped) > 3 else ""
                detail = f"｜跳过：{items}{more}"
            self._log(f"扫描 {DATA_DIR.name}\\：{scanned} 个文件{detail}")

        if self._used_builtin:
            self._log(f"data\\ 为空或无可读文件，已回退到内置演示语料（{len(self._chunks)} 条）")

        if self._chunks:
            per_source: dict[str, int] = {}
            for chunk in self._chunks:
                per_source[chunk.source] = per_source.get(chunk.source, 0) + 1
            dist = "  ".join(f"{name}({count})" for name, count in sorted(per_source.items()))
            self._log(f"片段 {len(self._chunks)} 条 ｜ {dist}")
        else:
            self._log("没有任何可用片段，检索会直接返回空")

    @property
    def size(self) -> int:
        """索引中的片段数量。"""
        return self._collection.count()

    @property
    def sources(self) -> list[str]:
        """当前语料涉及的来源文件（去重，便于自检）。"""
        return sorted({c.source for c in self._chunks})

    # ---------- 索引指纹：内容/embedding 任一变化都要重建 ----------
    def _embedding_identity(self) -> str:
        """换 embedding 等于换向量空间，索引必须重建。"""
        ef = self._embedding_function
        name = getattr(ef, "model_name", "") or getattr(ef, "name", "")
        if callable(name):          # chromadb 的 name() 是方法，不是属性
            try:
                name = name()
            except Exception:
                name = ""
        if not isinstance(name, str):
            name = ""
        return f"{type(ef).__name__}:{name}"

    def _fingerprint(self) -> str:
        h = hashlib.sha256()
        h.update(self._embedding_identity().encode("utf-8"))
        h.update(COLLECTION_NAME.encode("utf-8"))
        for chunk in self._chunks:
            h.update(f"{chunk.id}\x00{chunk.text}\x1f".encode("utf-8"))
        return h.hexdigest()[:16]

    def _manifest_path(self) -> Path:
        return self._index_dir / MANIFEST_NAME

    def _read_manifest(self) -> dict:
        try:
            return json.loads(self._manifest_path().read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError):
            return {}

    def _build_if_needed(self) -> None:
        """内容指纹、条数、embedding 三者都一致才跳过；否则重建（upsert 幂等）。

        旧实现只比条数：改了某条知识点的措辞但条数不变时，索引不会更新。
        """
        ids = [c.id for c in self._chunks]
        fingerprint = self._fingerprint()
        manifest = self._read_manifest()

        if manifest.get("fingerprint") == fingerprint and self.size == len(ids):
            self._log(f"指纹未变（{len(ids)} 条），跳过重建")
            return

        old_count = manifest.get("count")
        started = time.perf_counter()
        if ids:
            self._collection.upsert(
                ids=ids,
                documents=[c.text for c in self._chunks],
                metadatas=[{"source": c.source, "chunk_id": c.id} for c in self._chunks],
            )

        # 语料变短/删文件时清掉历史 id，避免检索到已删除的片段
        for stale in set(manifest.get("ids") or []) - set(ids):
            self._collection.delete(ids=[stale])

        self._manifest_path().write_text(
            json.dumps(
                {
                    "fingerprint": fingerprint,
                    "embedding": self._embedding_identity(),
                    "count": len(ids),
                    "ids": ids,
                    "sources": self.sources,
                    "documents_sha256": hashlib.sha256(
                        "\n".join(c.text for c in self._chunks).encode("utf-8")
                    ).hexdigest(),
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

        cost = time.perf_counter() - started
        if old_count is None:
            self._log(f"首次建库完成：{len(ids)} 条，耗时 {cost:.2f}s")
        else:
            self._log(f"内容有变化，已重建索引：{old_count} → {len(ids)} 条，耗时 {cost:.2f}s")

    def search(self, query: str, top_k: int = 3, hybrid: bool = True) -> list[Chunk]:
        """按语义相似度召回 Top-K 片段；空查询或空库返回空列表。

        hybrid=True 时先多召回一些候选，再用「向量分 + 中文词面覆盖率」重排，
        以补偿默认英文 embedding 在中文语料上的召回损失。
        """
        if not query.strip() or self.size == 0:
            return []

        pool = max(top_k, RERANK_POOL) if hybrid else top_k
        result = self._collection.query(
            query_texts=[query],
            n_results=min(pool, self.size),
        )
        documents = (result.get("documents") or [[]])[0]
        metadatas = (result.get("metadatas") or [[]])[0]
        distances = (result.get("distances") or [[]])[0]

        chunks: list[Chunk] = []
        for text, metadata, distance in zip(documents, metadatas, distances):
            # 集合使用 cosine 空间，distance = 1 - 余弦相似度
            score = max(0.0, min(1.0, 1.0 - float(distance)))
            if hybrid:
                score = (1 - LEXICAL_WEIGHT) * score + LEXICAL_WEIGHT * _lexical_score(
                    query, str(text)
                )
            meta = metadata or {}
            chunks.append(
                Chunk(
                    text=str(text),
                    # 引用编号优先用 文件名#段号，便于回指原文
                    source=str(meta.get("chunk_id") or meta.get("source") or "?"),
                    score=score,
                )
            )

        chunks.sort(key=lambda c: c.score, reverse=True)
        return chunks[:top_k]

    @staticmethod
    def format_context(chunks: list[Chunk]) -> str:
        """把召回片段拼成带来源标记的提示词上下文。"""
        return "\n".join(f"[{c.source}] {c.text}" for c in chunks)


@lru_cache(maxsize=1)
def get_knowledge_base() -> KnowledgeBase:
    """进程内单例，避免每次检索都重新加载索引与 embedding 模型。"""
    return KnowledgeBase()
