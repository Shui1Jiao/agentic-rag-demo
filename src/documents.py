"""知识库文档加载与切分：data/ 目录 -> 片段列表（每条带来源与段号）。

支持格式：.md / .txt / .docx / .pdf（后两者按需导入依赖，缺失时跳过该文件）。
切分策略：先按空行分段；段落超过 CHUNK_SIZE 再按字符窗口切并带重叠。
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
CHUNK_SIZE = 400
CHUNK_OVERLAP = 50
SUPPORTED = {".md", ".txt", ".docx", ".pdf"}


@dataclass(frozen=True)
class DocChunk:
    """一个可检索片段。id 形如 员工手册.md#03，可直接当引用编号。"""

    id: str
    source: str
    text: str


def _read_file(path: Path) -> str:
    suffix = path.suffix.lower()
    try:
        if suffix in {".md", ".txt"}:
            return path.read_text(encoding="utf-8", errors="ignore")
        if suffix == ".docx":
            from docx import Document  # 可选依赖
            # 用双换行连接，保留段落边界（否则整篇 docx 会被当成一段）
            return "\n\n".join(p.text for p in Document(str(path)).paragraphs if p.text.strip())
        if suffix == ".pdf":
            from pypdf import PdfReader  # 可选依赖
            return "\n\n".join((page.extract_text() or "") for page in PdfReader(str(path)).pages)
    except Exception as exc:  # 单个文件坏了不影响整体
        print(f"[documents] 跳过 {path.name}: {exc}")
    return ""


def split_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """先按空行分段，超长段落再按窗口切（带重叠）。"""
    paragraphs = [p.strip() for p in text.replace("\r\n", "\n").split("\n\n") if p.strip()]
    chunks: list[str] = []
    for para in paragraphs:
        if len(para) <= size:
            chunks.append(para)
            continue
        step = max(1, size - overlap)
        for i in range(0, len(para), step):
            piece = para[i : i + size].strip()
            if piece:
                chunks.append(piece)
    return chunks


def load_chunks(
    data_dir: Path | str = DATA_DIR,
    stats: dict | None = None,
) -> list[DocChunk]:
    """扫描目录加载所有支持的文件，返回片段列表；目录不存在返回空列表。

    stats 传入字典时回填统计信息，便于上层打印“扫了几个文件、跳过哪些、为什么”。
    """
    data_dir = Path(data_dir)
    if stats is None:
        stats = {}
    stats.setdefault("scanned", 0)
    stats.setdefault("loaded", [])
    stats.setdefault("skipped", [])
    stats.setdefault("dir_exists", data_dir.exists())

    if not data_dir.exists():
        return []

    out: list[DocChunk] = []
    for path in sorted(data_dir.rglob("*")):
        if not path.is_file():
            continue
        stats["scanned"] += 1
        if path.suffix.lower() not in SUPPORTED:
            stats["skipped"].append({"name": path.name, "reason": "格式不支持"})
            continue
        text = _read_file(path)
        if not text.strip():
            stats["skipped"].append({"name": path.name, "reason": "无可提取文本"})
            continue
        pieces = split_text(text)
        for i, piece in enumerate(pieces):
            out.append(DocChunk(id=f"{path.name}#{i:02d}", source=path.name, text=piece))
        stats["loaded"].append({"name": path.name, "chunks": len(pieces)})
    return out
