# Agentic‑RAG Demo
基于 LangGraph 实现的 Agentic‑RAG Agent。
Agent可以自主决策：知识库检索、工具调用（计算器、获取时间）、直接回答。

## ✨功能特性
- Agent自主思考决策路由
- RAG知识库问答
- Function‑Calling工具调用：加减乘除计算器、获取系统时间
- LangGraph状态管理、条件分支流转
- 控制台交互式对话

## 🛠环境依赖
Python >=3.10

## 📦安装
```bash
# 创建虚拟环境
python -m venv venv
venv\Scripts\activate

# 安装依赖
pip install .
