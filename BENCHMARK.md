# 实测数据（BENCHMARK）

- 实测日期：2026-10-10（第 2、3 节为 2026-11-01 在现版本上复跑，数值一致）
- 环境：Windows 11 / Python 3.11.6 / chromadb 1.5.9 / CPU 推理（无 GPU）
- 语料：评测固定用 `src/retriever.py` 的 `DOCUMENTS`，12 条知识点（`bench/retrieval_eval.py` 显式传入，避免 `data/` 变化影响可比性）
- 方法：全部为仓库内脚本可复跑，不含任何 LLM 调用（避免把模型侧延迟算成本项目性能）

## 1. 测试

```bash
python -m pytest tests -q
```

```
8 passed in 4.58s
```

现有 `tests/test_smoke.py` 覆盖：索引条数、检索相关性排序、空查询、切分边界、内容指纹重建、
judge 输出规整、条件边兜底等 8 个离线用例（不需要 API key）。

> 覆盖率数据（整体 53%、`retriever.py` 94%）是 2026-10-10 用 `python -m pytest tests -q --cov=src --cov-report=term-missing`
> 在 5 用例的旧版本上测的，保留作为当时基线；现在用例数已变为 8，覆盖率未重新测（需要额外装 `pytest-cov`）。

## 2. 检索质量与延迟（20 条人工标注 query）

评测集：20 条中文 query，人工标注每条应命中的语料条目标识 `kb#NN`；标注依据是"query 的语义中心落在哪条知识点"。
相关集合为人工真值，不是模型产物。

```bash
python bench/retrieval_eval.py baseline   # 纯向量
python bench/retrieval_eval.py hybrid     # 向量 + 中文词面重排
```

| 模式 | recall@3 | MRR@3 | 延迟 mean | p50 | p95 |
|---|---|---|---|---|---|
| 纯向量（baseline） | 13/20 = **65.0%** | 0.550 | 35.7 ms | 17.1 ms | 154.8 ms |
| 向量 + 词面重排 | 17/20 = **85.0%** | 0.825 | 42.9 ms | 18.3 ms | 172.5 ms |

（n=100 次计时/模式，两个模式各起独立进程，避免同进程顺序偏差）

**结论**：默认 embedding 是英文语料训练的 `all-MiniLM-L6-v2`，中文语义召回只有 65%；在向量分之上叠加一层
字符 bigram 覆盖率重排（不引入任何新模型或依赖），recall@3 提升到 **85%**。

## 3. 性能修复：embedding 会话被反复重建

**现象**：单次检索（12 条语料）耗时 p50 541 ms，明显不合理。

**定位**：`chromadb` 的 `DefaultEmbeddingFunction.__call__` 每次调用都会新建
`ONNXMiniLM_L6_V2()`（`chromadb/api/types.py:965`），而 ONNX 会话挂在实例的 `cached_property` 上 ——
等于每次检索都重新加载一遍模型。

**对照实验**（同一进程内各 11 次编码）：

| 调用方式 | 首次 | 后续均值 |
|---|---|---|
| `DefaultEmbeddingFunction()` | 278 ms | **155.6 ms** |
| 复用同一个 `ONNXMiniLM_L6_V2()` 实例 | 165 ms | **13.2 ms** |

**修复**：`src/retriever.py` 用 `lru_cache` 持有单例 embedding 实例，不再走每次新建的默认实现。

**效果**（同一评测脚本、同一评测集）：

| | 修复前 | 修复后 |
|---|---|---|
| 单次检索 p50 | 541.1 ms | **17.1 ms** |
| 单次检索 mean | 548.9 ms | **35.7 ms** |
| p95 | 776.7 ms | 154.8 ms |

> 注：修复后需删除 `.chroma/` 让索引按新 embedding 配置重建（`.chroma` 是派生数据，脚本会自动重跑建索引）。
