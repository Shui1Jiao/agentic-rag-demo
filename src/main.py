"""Agentic-RAG Demo：用 LangGraph 状态机实现「自主路由」的检索增强 Agent。

和普通 RAG 的区别：检索前多了一个 judge 节点，由大模型决定这次要走
检索（rag）、调用工具（tool）还是直接回答（direct），而不是无脑跑一条固定流水线。

运行方式（仓库根目录）：
    python -m src.main
"""

from __future__ import annotations

import os
from datetime import datetime
from typing import TypedDict

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph

try:
    from src.retriever import KnowledgeBase, get_knowledge_base
except ImportError:  # 兼容 python src/main.py 这种直接执行的方式
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from src.retriever import KnowledgeBase, get_knowledge_base

load_dotenv()

DEFAULT_MODEL = "deepseek-chat"
DEFAULT_BASE_URL = "https://api.deepseek.com/v1"
# judge 路由后进入知识库检索时召回的片段数
TOP_K = 3

# 合法路由决策。judge 的输出必须落在这一集合内，否则视为不可信并兜底。
VALID_ACTIONS = ("rag", "tool", "direct")


# ---------------------- 模型 ----------------------
_llm: ChatOpenAI | None = None


def get_llm() -> ChatOpenAI:
    """惰性初始化模型，这样导入模块和编译状态图都不需要 API key。"""
    global _llm
    if _llm is None:
        api_key = os.getenv("DEEPSEEK_API_KEY")
        if not api_key:
            raise SystemExit(
                "未找到 DEEPSEEK_API_KEY。请先复制 .env.example 为 .env 并填入密钥。"
            )
        _llm = ChatOpenAI(
            base_url=os.getenv("DEEPSEEK_BASE_URL", DEFAULT_BASE_URL),
            model=os.getenv("DEEPSEEK_MODEL", DEFAULT_MODEL),
            api_key=api_key,
            temperature=0,
        )
    return _llm


# ---------------------- 工具 ----------------------
def calculator(a: float, b: float, op: str) -> str:
    """计算器：add 加法 / sub 减法 / mul 乘法 / div 除法。"""
    if op == "add":
        res = a + b
    elif op == "sub":
        res = a - b
    elif op == "mul":
        res = a * b
    elif op == "div":
        if b == 0:
            return "错误：除数不能为 0"
        res = a / b
    else:
        return f"不支持运算符：{op}"
    return f"计算结果：{a} {op} {b} = {res}"


def get_current_time() -> str:
    return f"当前时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"


TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "数学加减乘除计算",
            "parameters": {
                "type": "object",
                "required": ["a", "b", "op"],
                "properties": {
                    "a": {"type": "number"},
                    "b": {"type": "number"},
                    "op": {"type": "string", "enum": ["add", "sub", "mul", "div"]},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_current_time",
            "description": "获取系统当前时间",
            "parameters": {"type": "object", "properties": {}},
        },
    },
]


def execute_tool(name: str, args: dict) -> str:
    """执行单个工具调用，参数不合法时返回可读的错误而不是抛异常。"""
    if name == "calculator":
        try:
            a = float(args["a"])
            b = float(args["b"])
            op = str(args["op"])
        except (KeyError, TypeError, ValueError) as exc:
            return f"[calculator 参数不合法] {exc}；实际收到：{args}"
        return calculator(a, b, op)
    if name == "get_current_time":
        return get_current_time()
    return f"[未知工具] {name}"


# ---------------------- Agent 状态 ----------------------
class AgentState(TypedDict, total=False):
    query: str
    thought: str
    action: str
    context: str
    citations: list[str]
    tool_calls: list[dict]
    tool_result: str
    tool_error: str
    warning: str
    final_answer: str


# ---------------------- 节点 ----------------------
JUDGE_PROMPT = """用户问题：{query}

选择处理方式，只输出一个单词：
rag：需要知识库（Agentic-RAG、LangGraph、RAG、Chroma、FAISS、Embedding、rerank、幻觉相关）
tool：需要数学计算，或需要获取当前时间
direct：普通问题，直接回答

只输出 rag / tool / direct 之一，不要输出任何其它文字。"""


def normalize_action(raw: str) -> str | None:
    """把模型输出规整成合法决策，识别不了就返回 None。"""
    token = raw.strip().strip("\"'` \t\r\n。.：:,，").lower()
    if token in VALID_ACTIONS:
        return token
    # 模型偶尔会多输出一句话，此时取其中第一个出现的合法动作词
    for action in VALID_ACTIONS:
        if action in token:
            return action
    return None


def judge_node(state: AgentState) -> AgentState:
    resp = get_llm().invoke(JUDGE_PROMPT.format(query=state["query"]))
    raw = resp.content if isinstance(resp.content, str) else str(resp.content)

    action = normalize_action(raw)
    warning = ""
    if action is None:
        # 关键：不允许静默降级。留着 warning，主循环会把原因打出来。
        action = "direct"
        warning = f"judge 返回了无法识别的决策 {raw!r}，已兜底为 direct"

    return {
        "thought": "judge 节点由 LLM 判断处理方式",
        "action": action,
        "context": "",
        "citations": [],
        "tool_calls": [],
        "tool_result": "",
        "tool_error": "",
        "warning": warning,
    }


def retrieve_node(state: AgentState) -> AgentState:
    kb = get_knowledge_base()
    chunks = kb.search(state["query"], top_k=TOP_K)
    if not chunks:
        return {
            "context": "",
            "citations": [],
            "warning": "知识库未召回任何片段",
        }
    return {
        "context": KnowledgeBase.format_context(chunks),
        "citations": [c.source for c in chunks],
    }


def tool_call_node(state: AgentState) -> AgentState:
    resp = get_llm().bind_tools(TOOL_SCHEMAS).invoke(
        [{"role": "user", "content": state["query"]}]
    )
    calls = list(getattr(resp, "tool_calls", None) or [])
    if not calls:
        return {
            "tool_calls": [],
            "tool_result": "",
            "tool_error": "模型没有选择调用任何工具",
        }

    outputs = []
    for call in calls:
        name = call.get("name", "")
        args = call.get("args") or {}
        outputs.append(f"{name}({args}) -> {execute_tool(name, args)}")

    return {
        "tool_calls": calls,
        "tool_result": "\n".join(outputs),
        "tool_error": "",
    }


def answer_node(state: AgentState) -> AgentState:
    action = state.get("action", "direct")

    if action == "rag":
        if not state.get("context"):
            # 宁可拒答，也不拿空上下文硬编
            return {"final_answer": "知识库中没有检索到相关内容，无法基于知识库回答这个问题。"}
        prompt = f"""严格依据下面的知识库片段回答问题，禁止编造。

知识库片段：
{state["context"]}

问题：{state["query"]}

要求：只使用上述片段中的信息；如果片段不足以回答，直接说明「知识库中没有足够信息」。
在结论后用 [来源: 片段编号] 标注依据，例如 [来源: kb#03]。"""

    elif action == "tool":
        if not state.get("tool_result"):
            reason = state.get("tool_error") or "未知原因"
            return {"final_answer": f"工具调用没有返回可用结果（{reason}），无法给出可靠答案。"}
        prompt = f"""工具执行结果：
{state["tool_result"]}

请基于工具结果回答用户问题：{state["query"]}"""

    else:
        prompt = f"直接回答用户问题：{state['query']}"

    resp = get_llm().invoke(prompt)
    return {"final_answer": resp.content}


def route_func(state: AgentState) -> str:
    """条件边：把 judge 的决策映射到下一个节点，未知值一律走 answer。"""
    return {
        "rag": "retrieve",
        "tool": "tool_call",
    }.get(state.get("action", ""), "answer")


# ---------------------- 构建状态图 ----------------------
def build_graph():
    workflow = StateGraph(AgentState)
    workflow.add_node("judge", judge_node)
    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("tool_call", tool_call_node)
    workflow.add_node("answer", answer_node)

    workflow.add_edge(START, "judge")
    workflow.add_conditional_edges(
        "judge",
        route_func,
        {"retrieve": "retrieve", "tool_call": "tool_call", "answer": "answer"},
    )
    workflow.add_edge("retrieve", "answer")
    workflow.add_edge("tool_call", "answer")
    workflow.add_edge("answer", END)

    return workflow.compile()


app = build_graph()


def print_mermaid_graph() -> None:
    print("\n==== 状态图（Mermaid）====")
    print(app.get_graph().draw_mermaid())


def warm_up_knowledge_base() -> None:
    print("正在初始化向量库（首次运行需下载 embedding 模型，约 80MB）...")
    kb = get_knowledge_base()
    print(f"知识库就绪，共 {kb.size} 个片段。")


# ---------------------- 主入口 ----------------------
def run_once(query: str) -> AgentState:
    """执行一轮问答，返回完整状态。"""
    result = app.invoke({"query": query})
    if result.get("warning"):
        print(f"warning：{result['warning']}")
    print(f"thought：{result.get('thought', '')}")
    print(f"action：{result.get('action', '')}")
    if result.get("citations"):
        print(f"引用来源：{', '.join(result['citations'])}")
    if result.get("context"):
        print(f"检索上下文：\n{result['context']}")
    if result.get("tool_result"):
        print(f"工具结果：{result['tool_result']}")
    print(f"\n【Agent 回答】{result.get('final_answer', '')}")
    return result


def main() -> None:
    print_mermaid_graph()
    warm_up_knowledge_base()
    print("\n==== Agentic-RAG，输入 exit 退出 ====")

    while True:
        try:
            user_input = input("\n请输入问题：")
        except (EOFError, KeyboardInterrupt):
            print("\n退出程序")
            return

        if user_input.strip().lower() in {"exit", "quit", "q"}:
            print("退出程序")
            return
        if not user_input.strip():
            continue

        run_once(user_input)


if __name__ == "__main__":
    main()
