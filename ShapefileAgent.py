from dotenv import load_dotenv
import getpass
import os
import functools
import json
import operator
from typing import Annotated, Sequence, TypedDict
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_experimental.utilities import PythonREPL
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import BaseMessage, FunctionMessage, HumanMessage, SystemMessage
from langchain.tools.render import format_tool_to_openai_function
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.graph import END, StateGraph
from langgraph.prebuilt.tool_executor import ToolExecutor, ToolInvocation


# --- 1. 환경 변수 및 API 키 설정 ---
load_dotenv(); _set_if_undefined("OPENAI_API_KEY"); _set_if_undefined("TAVILY_API_KEY"); _set_if_undefined("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = "true"; os.environ["LANGCHAIN_PROJECT"] = "Shapefile-Agent-Optimized"


# --- 2. 도구 정의 ---
web_search_tool = TavilySearchResults(max_results=3, name="web_search")
repl = PythonREPL()
@tool
def python_repl(code: Annotated[str, "The python code to execute."]):
    """Executes Python code in a REPL environment to process Shapefiles."""
    try: result = repl.run(code)
    except BaseException as e: return f"Failed to execute. Error: {repr(e)}"
    return f"Successfully executed:\n```python\n{code}\n```\nStdout: {result}"


# --- 3. 에이전트 상태, 생성자, 노드 정의 ---
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    sender: str
    next_node: str
    max_messages: int 

llm = ChatOpenAI(model="gpt-4o")

def create_agent(agent_llm, tools, system_message: str):
    functions = [format_tool_to_openai_function(t) for t in tools]
    prompt = ChatPromptTemplate.from_messages([
        MessagesPlaceholder(variable_name="messages"),
    ])
    return prompt | agent_llm.bind_functions(functions)

def agent_node(state, agent, name, system_message_str):
    max_messages = state.get("max_messages", 10)
    pruned_messages = [SystemMessage(content=system_message_str)] + state['messages'][-max_messages:]
    
    pruned_state = state.copy()
    pruned_state["messages"] = pruned_messages
    
    result = agent.invoke(pruned_state)

    if not isinstance(result, FunctionMessage):
        result = HumanMessage(**result.dict(exclude={"type", "name"}), name=name)
    
    next_node = ""
    if name == "Supervisor" and "NEXT:" in result.content:
        next_node = result.content.split("NEXT:")[-1].strip()

    return {"messages": [result], "sender": name, "next_node": next_node}


# --- 4. 에이전트 정의 (강화된 프롬프트 적용) ---
# 4-1. Supervisor Agent
supervisor_system_message = (
    "You are a Supervisor agent managing a team of GIS specialists. Your **only** role is to analyze user requests, break them down into serial subtasks, and delegate them. "
    "You **MUST NOT** write or execute Python code yourself. You **MUST NOT** perform any task directly. "
    "Delegate each subtask to **one** of the following specialists: **DataLoader, AttributeUpdater, AreaExpander, BlockCreator, RoadModifier, RoadCreator, FileConverter, WebSearcher.** "
    "After a worker completes a task, review the result and delegate the next step. "
    "To delegate, you **MUST** end your response with 'NEXT: [Worker Agent Name]'. "
    "If a worker's output is incorrect or insufficient, re-instruct them or delegate to a different worker. "
    "Once all steps are complete and the user's request is fully satisfied, prefix your response with 'FINAL ANSWER'."
)
supervisor_agent = create_agent(llm, [web_search_tool], supervisor_system_message)
supervisor_node = functools.partial(agent_node, agent=supervisor_agent, name="Supervisor", system_message_str=supervisor_system_message)

# 4-2. Worker Agents
worker_prompts = {
    "DataLoader": "You are a GIS specialist named DataLoader. Your **only** task is to load a Shapefile using GeoPandas. Write and execute the necessary python code. **Do not perform any other actions.**",
    "AttributeUpdater": "You are a GIS specialist named AttributeUpdater. Your **only** task is to update attributes in a Shapefile. Write and execute the code. **Do not perform any other actions.**",
    "AreaExpander": "You are a GIS specialist named AreaExpander. Your **only** task is to expand the area of specified elements in a Shapefile. Write and execute the code. **Do not perform any other actions.**",
    "BlockCreator": "You are a GIS specialist named BlockCreator. Your **only** task is to create new block elements. This includes adding to an existing map or creating from scratch on a blank map. Write and execute the code. **Do not perform any other actions.**",
    "RoadModifier": "You are a GIS specialist named RoadModifier. Your **only** task is to modify existing roads in a Shapefile. Write and execute the code. **Do not perform any other actions.**",
    "RoadCreator": "You are a GIS specialist named RoadCreator. Your **only** task is to create new roads in a Shapefile. Write and execute the code. **Do not perform any other actions.**",
    "FileConverter": "You are a GIS specialist named FileConverter. Your **only** task is to convert a Shapefile to another format. Write and execute the code. **Do not perform any other actions.**",
    "WebSearcher": "You are a specialist named WebSearcher. Your **only** task is to use the `web_search` tool to find external information and provide it to the Supervisor. **Do not answer questions yourself.**"
}

worker_nodes = {}
for name, sys_prompt in worker_prompts.items():
    tools = [web_search_tool] if name == "WebSearcher" else [python_repl]
    agent = create_agent(llm, tools, sys_prompt)
    worker_nodes[name] = functools.partial(agent_node, agent=agent, name=name, system_message_str=sys_prompt)


# --- 5. 도구 및 라우팅 로직 정의 ---
tool_executor = ToolExecutor([web_search_tool, python_repl])

def tool_node(state: AgentState) -> dict:
    last_message = state["messages"][-1]
    tool_call = last_message.additional_kwargs["function_call"]
    action = ToolInvocation(tool=tool_call["name"], tool_input=json.loads(tool_call["arguments"]))
    response = tool_executor.invoke(action)
    return {"messages": [FunctionMessage(content=str(response), name=action.tool)]}

def main_router(state: AgentState) -> str:
    last_message = state["messages"][-1]
    if "function_call" in last_message.additional_kwargs: return "call_tool"
    if "FINAL ANSWER" in last_message.content: return "end"
    if state.get("next_node"): return state["next_node"]
    return "Supervisor"


# --- 6. 그래프 구성 및 컴파일 ---
workflow = StateGraph(AgentState)
workflow.add_node("Supervisor", supervisor_node)
for name, node in worker_nodes.items(): workflow.add_node(name, node)
workflow.add_node("call_tool", tool_node)
workflow.set_entry_point("Supervisor")

supervisor_edges = {name: name for name in worker_prompts.keys()}
supervisor_edges.update({"call_tool": "call_tool", "end": END})
workflow.add_conditional_edges("Supervisor", main_router, supervisor_edges)

for name in worker_prompts.keys():
    workflow.add_conditional_edges(name, main_router, {"Supervisor": "Supervisor", "call_tool": "call_tool"})

tool_edges = {name: name for name in worker_prompts.keys()}
tool_edges["Supervisor"] = "Supervisor"
workflow.add_conditional_edges("call_tool", lambda x: x["sender"], tool_edges)

graph = workflow.compile()


# --- 7. 그래프 실행 ---
if __name__ == '__main__':
    user_input = input("메시지를 입력하세요: ")
    initial_state = {"messages": [HumanMessage(content=user_input)], "max_messages": 10}
    
    for s in graph.stream(initial_state, {"recursion_limit": 200}):
        for key, value in s.items():
            print(f"--- Node: '{key}' ---")
            print(value)
        print("\n=====================\n")