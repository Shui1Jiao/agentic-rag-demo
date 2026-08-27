import os
from datetime import datetime
from typing import TypedDict
from dotenv import load_dotenv
from langgraph.graph import StateGraph
from langchain_openai import ChatOpenAI

# 加载环境变量
load_dotenv()
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")

llm = ChatOpenAI(
    base_url="https://api.deepseek.com/v1",
    model="deepseek-chat",  # ❌旧名字，已经失效
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    temperature=0
)


# ----------------------工具函数----------------------
def calculator(a: float, b: float, op: str) -> str:
    """
    计算器：add加法 sub减法 mul乘法 div除法
    """
    if op == "add":
        res = a + b
    elif op == "sub":
        res = a - b
    elif op == "mul":
        res = a * b
    elif op == "div":
        if b == 0:
            return "错误：除数不能为0"
        res = a / b
    else:
        return "不支持运算符"
    return f"计算结果：{a} {op} {b} = {res}"

def get_current_time() -> str:
    return f"当前时间：{datetime.now().strftime('%Y‑%m‑%d %H:%M:%S')}"

tools = [
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
                    "op": {"type": "string", "enum": ["add","sub","mul","div"]}
                }
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_current_time",
            "description": "获取系统当前时间",
            "parameters": {"type": "object", "properties": {}}
        }
    }
]

# ----------------------模拟知识库----------------------
knowledge_base = """
Agentic‑RAG：智能检索增强生成，Agent自主判断是否检索知识库，是否调用工具。
LangGraph：LangChain官方Agent开发框架，核心是状态管理、条件分支、循环。
RAG：检索增强生成，检索外部知识交给大模型生成答案。
Chroma：轻量级本地向量数据库。
Function‑Calling：大模型输出参数调用自定义工具。
"""

# ----------------------Agent状态定义----------------------
class AgentState(TypedDict):
    query: str
    thought: str
    action: str
    context: str
    tool_call_info: dict
    tool_result: str
    final_answer: str

# ----------------------节点定义----------------------
def judge_node(state: AgentState):
    prompt = f"""用户问题：{state['query']}
选择处理方式：
rag：需要知识库（Agentic‑RAG、LangGraph、RAG、Chroma相关）
tool：数学计算、获取时间
direct：普通问题直接回答
只输出单词 rag / tool / direct，不要多余文字。
"""
    resp = llm.invoke(prompt)
    act = resp.content.strip()
    return {
        "thought": "Agent判断处理方式",
        "action": act,
        "context": "",
        "tool_call_info": {},
        "tool_result": ""
    }

def retrieve_node(state: AgentState):
    return {"context": knowledge_base}

def tool_call_node(state: AgentState):
    messages = [{"role":"user","content": state["query"]}]
    resp = llm.bind_tools(tools).invoke(messages)
    tool_result = ""
    if resp.tool_calls:
        tc = resp.tool_calls[0]
        fname = tc["name"]
        args = tc["args"]
        if fname == "calculator":
            tool_result = calculator(a=args["a"], b=args["b"], op=args["op"])
        elif fname == "get_current_time":
            tool_result = get_current_time()
    return {"tool_call_info": resp.tool_calls, "tool_result": tool_result}

def answer_node(state: AgentState):
    if state["action"] == "rag":
        prompt = f"""参考知识库回答，禁止编造。
知识库：
{state['context']}
问题：{state['query']}
"""
    elif state["action"] == "tool":
        prompt = f"""工具结果：
{state['tool_result']}
基于工具结果回答用户问题：{state['query']}
"""
    else:
        prompt = f"直接回答用户问题：{state['query']}"
    resp = llm.invoke(prompt)
    return {"final_answer": resp.content}

def route_func(state):
    act = state["action"]
    if act == "rag":
        return "retrieve"
    elif act == "tool":
        return "tool_call"
    else:
        return "answer"

# ----------------------构建图----------------------
workflow = StateGraph(AgentState)
workflow.add_node("judge", judge_node)
workflow.add_node("retrieve", retrieve_node)
workflow.add_node("tool_call", tool_call_node)
workflow.add_node("answer", answer_node)

workflow.set_entry_point("judge")
workflow.add_conditional_edges("judge", route_func)
workflow.add_edge("retrieve", "answer")
workflow.add_edge("tool_call", "answer")
workflow.set_finish_point("answer")

app = workflow.compile()

# 打印mermaid流程图代码
def print_mermaid_graph():
    print("\n==== Mermaid流程图代码 ====")
    print(app.get_graph().draw_mermaid())

# ----------------------主入口----------------------
def main():
    print_mermaid_graph()
    print("\n==== Agentic‑RAG Agent，输入exit退出 ====")
    while True:
        user_input = input("\n请输入问题：")
        if user_input.strip().lower() == "exit":
            print("退出程序")
            break
        res = app.invoke({"query": user_input})
        print(f"thought：{res['thought']}")
        print(f"action：{res['action']}")
        print(f"检索上下文：{res.get('context','')}")
        print(f"工具结果：{res.get('tool_result','')}")
        print(f"\n【Agent回答】{res['final_answer']}")

if __name__ == "__main__":
    main()
