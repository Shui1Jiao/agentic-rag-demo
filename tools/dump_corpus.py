"""把知识库里的全部语料导出来看：data/ 目录（或内置兜底语料）的每一条片段。

用法（在项目根目录）：
    python tools/dump_corpus.py             # 生成 corpus.txt（人读）+ corpus.json（机读）
    python tools/dump_corpus.py --stdout    # 直接打印到终端（需 UTF-8 控制台）
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.retriever import get_documents  # noqa: E402


def build_report(chunks):
    lines: list[str] = []
    per_source = Counter(c.source for c in chunks)
    total_chars = sum(len(c.text) for c in chunks)

    lines.append("=" * 60)
    lines.append(f"语料总数：{len(chunks)} 条片段 ｜ 来源文件：{len(per_source)} 个 ｜ 总字符：{total_chars}")
    lines.append("=" * 60)
    lines.append("")
    lines.append("【按来源统计】")
    for source, count in sorted(per_source.items()):
        chars = sum(len(c.text) for c in chunks if c.source == source)
        lines.append(f"  {source:<24} {count:>3} 条  {chars:>6} 字")
    lines.append("")
    lines.append("【逐条明细】")
    for i, chunk in enumerate(chunks, 1):
        lines.append("-" * 60)
        lines.append(f"[{i:02d}] {chunk.id}｜{len(chunk.text)} 字")
        lines.append(chunk.text)
    lines.append("-" * 60)
    return "\n".join(lines), per_source, total_chars


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stdout", action="store_true", help="直接打印，不写文件")
    ap.add_argument("--out", default="corpus", help="输出文件名前缀（默认 corpus）")
    args = ap.parse_args()

    chunks = list(get_documents())
    report, per_source, total_chars = build_report(chunks)

    if args.stdout:
        print(report)
        return

    txt_path = ROOT / f"{args.out}.txt"
    json_path = ROOT / f"{args.out}.json"
    txt_path.write_text(report, encoding="utf-8")
    json_path.write_text(
        json.dumps(
            {
                "total_chunks": len(chunks),
                "total_chars": total_chars,
                "sources": dict(sorted(per_source.items())),
                "chunks": [
                    {"id": c.id, "source": c.source, "chars": len(c.text), "text": c.text}
                    for c in chunks
                ],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"已导出 {len(chunks)} 条片段 -> {txt_path.name} / {json_path.name}")
    print(f"（用编辑器打开 {txt_path} 查看全文）")


if __name__ == "__main__":
    main()
