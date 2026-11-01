"""导出 LangGraph 状态图：graph.mmd（离线，纯文本）+ graph.png（走 mermaid.ink，需联网）。

用法：python tools/export_graph.py
"""
from __future__ import annotations

import base64
import io
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.main import app  # noqa: E402  导入只编译图，不需要 API key


def main() -> None:
    mermaid = app.get_graph().draw_mermaid()
    mmd = ROOT / "graph.mmd"
    mmd.write_text(mermaid, encoding="utf-8")
    print(f"已写出 {mmd}（可直接贴到 https://mermaid.live 出图）")

    token = base64.urlsafe_b64encode(mermaid.encode()).decode().rstrip("=")
    url = f"https://mermaid.ink/img/{token}?bgColor=white&type=png"
    png = ROOT / "graph.png"
    try:
        with urllib.request.urlopen(url, timeout=30) as resp:
            png.write_bytes(resp.read())
        print(f"已写出 {png}（{png.stat().st_size} 字节）")
    except Exception as exc:  # 离线时不阻塞
        print(f"PNG 生成失败（可能无网络）：{exc}")


if __name__ == "__main__":
    main()
