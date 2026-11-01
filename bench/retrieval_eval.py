"""检索层评测：recall@3 / MRR@3 + 检索延迟（全程本地，不含任何 LLM 调用）。

评测集说明：20 条中文 query 由人工按语料（src/retriever.py 的 DOCUMENTS）编写，
每条标注它应命中的语料条目标识 kb#NN，标注依据是「该 query 的语义中心落在哪条知识点」。
相关集合是人工真值，不是模型产物；脚本、语料、评测集都在仓库里，可原样复跑。

运行：
    python bench/retrieval_eval.py
"""

from __future__ import annotations

import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.retriever import DOCUMENTS, KnowledgeBase, builtin_chunks  # noqa: E402

TOP_K = 3
REPEATS = 5

# (query, 相关的 kb#NN 集合)
EVAL_SET: list[tuple[str, set[str]]] = [
    ("Agentic-RAG 和普通 RAG 有什么区别", {"kb#00"}),
    ("RAG 是什么，为什么能缓解知识过时", {"kb#01"}),
    ("LangGraph 用什么来组织流程", {"kb#02"}),
    ("状态图里的节点是什么东西", {"kb#03"}),
    ("怎么根据当前状态决定下一个走哪个节点", {"kb#04"}),
    ("模型怎么调用外部函数并把结果回传", {"kb#05"}),
    ("常见的向量数据库都有哪些", {"kb#06"}),
    ("Chroma 支持哪些运行模式", {"kb#07"}),
    ("FAISS 自带 embedding 模型吗", {"kb#08"}),
    ("文本是怎么变成向量的", {"kb#09"}),
    ("召回之后怎么进一步提升相关性", {"kb#10"}),
    ("大模型胡说八道怎么治", {"kb#11"}),
    ("条件边是干什么用的", {"kb#04"}),
    ("把文本存起来做相似度检索该用什么", {"kb#06"}),
    ("轻量级、适合原型的本地向量库", {"kb#07"}),
    ("只做索引和检索、需要自己给向量的库", {"kb#08"}),
    ("语义相近的文本在向量空间里距离更近", {"kb#09"}),
    ("rerank 的作用是什么", {"kb#10"}),
    ("模型输出结构化的工具名与参数", {"kb#05"}),
    ("状态图容器叫什么", {"kb#03"}),
]


def evaluate(kb: KnowledgeBase, label: str, hybrid: bool) -> tuple[float, list[str], list[float]]:
    hits = 0
    reciprocal_ranks: list[float] = []
    latencies_ms: list[float] = []
    misses: list[str] = []

    for query, relevant in EVAL_SET:
        got = [c.source for c in kb.search(query, top_k=TOP_K, hybrid=hybrid)]
        rank = next((i + 1 for i, s in enumerate(got) if s in relevant), None)
        if rank:
            hits += 1
            reciprocal_ranks.append(1.0 / rank)
        else:
            reciprocal_ranks.append(0.0)
            misses.append(f"{query} -> {got}（期望 {sorted(relevant)}）")

        for _ in range(REPEATS):
            t0 = time.perf_counter()
            kb.search(query, top_k=TOP_K, hybrid=hybrid)
            latencies_ms.append((time.perf_counter() - t0) * 1000)

    n = len(EVAL_SET)
    latencies_ms.sort()
    recall = hits / n
    print(f"{label:<18} recall@{TOP_K}={hits}/{n}={recall * 100:.1f}%  "
          f"MRR={sum(reciprocal_ranks) / n:.3f}  "
          f"延迟 mean={statistics.mean(latencies_ms):.1f}ms "
          f"p50={latencies_ms[len(latencies_ms) // 2]:.1f}ms "
          f"p95={latencies_ms[int(len(latencies_ms) * 0.95)]:.1f}ms "
          f"max={latencies_ms[-1]:.1f}ms  (n={len(latencies_ms)})")
    return recall, misses, latencies_ms


def main() -> None:
    # 两种模式各起一个进程跑，避免同进程先后执行带来的顺序偏差
    mode = sys.argv[1] if len(sys.argv) > 1 else "both"
    # 评测固定在 12 条内置语料上，保证与 BENCHMARK 的口径可比
    kb = KnowledgeBase(chunks=builtin_chunks())
    kb.search("预热查询", top_k=TOP_K)  # 首次会加载 ONNX 模型并建索引，不计入统计

    print(f"语料条数 {kb.size}（DOCUMENTS={len(DOCUMENTS)}）｜评测集 {len(EVAL_SET)} 条｜每模式 {REPEATS} 次计时")
    if mode in ("baseline", "both"):
        _, misses_baseline, _ = evaluate(kb, "纯向量(baseline)", hybrid=False)
        print("baseline 未命中：")
        for m in misses_baseline:
            print("  -", m)
    if mode in ("hybrid", "both"):
        _, misses_hybrid, _ = evaluate(kb, "向量+词面重排", hybrid=True)
        print("混合检索 未命中：")
        for m in misses_hybrid:
            print("  -", m)


if __name__ == "__main__":
    main()
