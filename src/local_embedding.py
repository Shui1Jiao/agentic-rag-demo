"""项目内自带 embedding 模型，避免首次运行联网下载 86MB。

模型放在 项目根/models/all-MiniLM-L6-v2/onnx/ 下（与 chromadb 缓存结构一致），
通过覆盖 DOWNLOAD_PATH 让 chromadb 认为"模型已存在"，从而完全跳过下载。
"""
from __future__ import annotations

from pathlib import Path

from chromadb.utils.embedding_functions.onnx_mini_lm_l6_v2 import ONNXMiniLM_L6_V2

PROJECT_MODEL_DIR = Path(__file__).resolve().parent.parent / "models" / "all-MiniLM-L6-v2"


class ProjectLocalONNX(ONNXMiniLM_L6_V2):
    """把模型目录固定到项目内，不读写用户缓存、不联网。"""

    DOWNLOAD_PATH = PROJECT_MODEL_DIR


def model_is_present() -> bool:
    return (PROJECT_MODEL_DIR / "onnx" / "model.onnx").exists()
