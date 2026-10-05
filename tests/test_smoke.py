"""检索层冒烟测试，不需要 API key。

运行：python -m pytest tests -q
首次运行会下载 Chroma 默认的 embedding 模型（约 80MB）。
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.main import normalize_action, route_func
from src.retriever import DOCUMENTS, KnowledgeBase


def make_kb(tmp_path) -> KnowledgeBase:
    return KnowledgeBase(index_dir=tmp_path / "chroma")


def test_index_contains_every_document(tmp_path):
    kb = make_kb(tmp_path)
    assert kb.size == len(DOCUMENTS)


def test_search_returns_relevant_chunk(tmp_path):
    kb = make_kb(tmp_path)
    chunks = kb.search("Chroma 和 FAISS 有什么区别", top_k=3)

    assert chunks, "应至少召回一条片段"
    assert all(0.0 <= c.score <= 1.0 for c in chunks)
    assert any("FAISS" in c.text or "Chroma" in c.text for c in chunks)
    # 结果按相似度降序
    scores = [c.score for c in chunks]
    assert scores == sorted(scores, reverse=True)


def test_empty_query_returns_nothing(tmp_path):
    kb = make_kb(tmp_path)
    assert kb.search("   ") == []


def test_normalize_action_accepts_noisy_output():
    assert normalize_action("rag") == "rag"
    assert normalize_action("  TOOL.\n") == "tool"
    assert normalize_action('"direct"') == "direct"
    assert normalize_action("我觉得应该用 rag 来处理") == "rag"
    assert normalize_action("???") is None


def test_route_func_falls_back_to_answer():
    assert route_func({"action": "rag"}) == "retrieve"
    assert route_func({"action": "tool"}) == "tool_call"
    assert route_func({"action": "direct"}) == "answer"
    assert route_func({"action": "garbage"}) == "answer"
    assert route_func({}) == "answer"
