"""冒烟测试：不需要 API key。

运行：python -m pytest tests -q
首次运行会用到项目内 models/ 的 embedding 模型（离线）。
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.documents import DocChunk, load_chunks, split_text
from src.main import normalize_action, route_func
from src.retriever import KnowledgeBase, builtin_chunks, get_documents


def make_kb(tmp_path, chunks=None) -> KnowledgeBase:
    # verbose=False：测试里不要刷启动日志
    return KnowledgeBase(index_dir=tmp_path / "chroma", chunks=chunks, verbose=False)


def test_index_contains_every_chunk(tmp_path):
    chunks = get_documents()
    kb = make_kb(tmp_path, chunks)
    assert kb.size == len(chunks)


def test_search_returns_relevant_chunk(tmp_path):
    kb = make_kb(tmp_path, builtin_chunks())
    chunks = kb.search("Chroma 和 FAISS 有什么区别", top_k=3)

    assert chunks, "应至少召回一条片段"
    assert all(0.0 <= c.score <= 1.0 for c in chunks)
    assert any("FAISS" in c.text or "Chroma" in c.text for c in chunks)
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


def test_rebuild_when_chunk_content_changed(tmp_path):
    """条数不变但内容改了，索引必须重建（旧实现只比条数，这里会失败）。"""
    base = builtin_chunks()
    kb1 = make_kb(tmp_path, base)
    assert any("Chroma：轻量级本地向量数据库" in c.text for c in kb1.search("Chroma", top_k=6))

    changed = list(base)
    changed[7] = DocChunk(id=changed[7].id, source=changed[7].source,
                          text="Chroma：轻量级本地向量数据库（2026 改版说明）。")
    kb2 = make_kb(tmp_path, changed)
    hits = kb2.search("Chroma 改版", top_k=6)
    assert kb2.size == len(changed)
    assert any("2026 改版说明" in c.text for c in hits), "内容改了但索引没重建"


def test_split_text_chunks_long_paragraph():
    long_para = "甲" * 1000
    pieces = split_text(long_para, size=400, overlap=50)
    assert len(pieces) >= 3
    assert all(len(p) <= 400 for p in pieces)


def test_load_chunks_reads_new_file_and_drops_deleted(tmp_path):
    """新增文件能被检索到；删除后不再返回。"""
    data = tmp_path / "data"
    data.mkdir()
    (data / "制度.md").write_text("年假规则：满一年 5 天，满三年 10 天。\n\n报销：10 个工作日内提交。", encoding="utf-8")
    chunks = load_chunks(data)
    assert chunks and all(c.source == "制度.md" for c in chunks)
    assert chunks[0].id.startswith("制度.md#")

    kb = make_kb(tmp_path, chunks)
    assert any("年假" in c.text for c in kb.search("年假有几天", top_k=3))

    (data / "制度.md").unlink()
    assert load_chunks(data) == []

    kb2 = make_kb(tmp_path, load_chunks(data))
    assert kb2.size == 0
    assert kb2.search("年假有几天") == []
