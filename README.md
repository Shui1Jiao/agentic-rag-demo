# Agentic‑RAG Demo
基于 LangGraph + DeepSeek 实现的智能Agentic‑RAG演示项目。
实现Agent自主判断：知识库检索 / 工具函数调用 / 直接回答，支持计算器工具调用。

## 项目架构
使用LangGraph构建状态图，节点：judge 判断节点 → retrieve知识库检索 / tool_call工具调用 → answer输出回答。
- judge：判断用户问题，选择分支：检索知识库 / 调用工具 / 直接回答
- retrieve：RAG知识库检索
- tool_call：执行自定义工具（计算器）
- answer：整理输出最终答案

## 环境依赖
```bash
pip install langgraph langchain-openai python-dotenv
