from pydantic import BaseModel, Field
from typing import Annotated, List, Any
from operator import add
from api.agents.agents import RAGUsedContext, product_qna_agent, shopping_cart_agent, warehouse_manager_agent, coordinator_agent, Delegation
from api.agents.tools import get_formatted_item_context, get_formatted_reviews_context, get_shopping_cart, get_shopping_cart_for_sse, remove_from_cart, add_to_shopping_cart, check_warehouse_availability, reserve_warehouse_items
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue
from langgraph.checkpoint.postgres import PostgresSaver
import json


class AgentProperties(BaseModel):
    iteration: int = 0
    final_answer: bool = False

class CoordinatorAgentProperties(BaseModel):
    iteration: int = 0
    final_answer: bool = False
    plan: List[Delegation] = []
    next_agent: str = ""

class State(BaseModel):
    messages: Annotated[List[Any], add] = []
    user_intent: str = ""
    product_qna_agent: AgentProperties = AgentProperties()
    shopping_cart_agent: AgentProperties = AgentProperties()
    warehouse_manager_agent: AgentProperties = AgentProperties()
    coordinator_agent: CoordinatorAgentProperties = CoordinatorAgentProperties()
    answer: str = ""
    references: list[RAGUsedContext] = []
    user_id: str = ""
    cart_id: str = ""
    trace_id: str = ""
    
### Edges

def product_qna_agent_tool_router(state) -> str:

    if state.product_qna_agent.final_answer:
        return "end"
    elif state.product_qna_agent.iteration > 5:
        return "end"
    elif len(state.messages[-1].tool_calls) > 0:
        return "tools"
    else:
        return "end"

    
def shopping_cart_agent_tool_router(state) -> str:

    if state.shopping_cart_agent.final_answer:
        return "end"
    elif state.shopping_cart_agent.iteration > 4:
        return "end"
    elif len(state.messages[-1].tool_calls) > 0:
        return "tools"
    else:
        return "end"

    
def warehouse_manager_agent_tool_router(state) -> str:

    if state.warehouse_manager_agent.final_answer:
        return "end"
    elif state.warehouse_manager_agent.iteration > 4:
        return "end"
    elif len(state.messages[-1].tool_calls) > 0:
        return "tools"
    else:
        return "end"

    
def coordinator_agent_edge(state) -> str:

    if state.coordinator_agent.final_answer:
        return "end"
    elif state.coordinator_agent.iteration > 4:
        return "end"
    elif state.coordinator_agent.next_agent == "product_qna_agent":
        return "product_qna_agent"
    elif state.coordinator_agent.next_agent == "shopping_cart_agent":
        return "shopping_cart_agent"
    elif state.coordinator_agent.next_agent == "warehouse_manager_agent":
        return "warehouse_manager_agent"
    else:
        return "end"
    

### Workflow

workflow = StateGraph(State)

product_qna_agent_tools = [get_formatted_item_context, get_formatted_reviews_context]
product_qna_agent_tool_node = ToolNode(product_qna_agent_tools)

shopping_cart_agent_tools = [get_shopping_cart, remove_from_cart, add_to_shopping_cart]
shopping_cart_agent_tool_node = ToolNode(shopping_cart_agent_tools)

warehouse_manager_agent_tools = [check_warehouse_availability, reserve_warehouse_items]
warehouse_manager_agent_tool_node = ToolNode(warehouse_manager_agent_tools)

workflow.add_node("product_qna_agent_tool_node", product_qna_agent_tool_node)
workflow.add_node("shopping_cart_agent_tool_node", shopping_cart_agent_tool_node)
workflow.add_node("warehouse_manager_agent_tool_node", warehouse_manager_agent_tool_node)
workflow.add_node("product_qna_agent", product_qna_agent)
workflow.add_node("shopping_cart_agent", shopping_cart_agent)
workflow.add_node("warehouse_manager_agent", warehouse_manager_agent)
workflow.add_node("coordinator_agent", coordinator_agent)

workflow.add_edge(START, "coordinator_agent")

workflow.add_conditional_edges(
    "coordinator_agent",
    coordinator_agent_edge,
    {
        "product_qna_agent": "product_qna_agent",
        "shopping_cart_agent": "shopping_cart_agent",
        "warehouse_manager_agent": "warehouse_manager_agent",
        "end": END
    }
)

workflow.add_conditional_edges(
    "product_qna_agent",
    product_qna_agent_tool_router,
    {
        "tools": "product_qna_agent_tool_node",
        "end": "coordinator_agent"
    }
)

workflow.add_conditional_edges(
    "shopping_cart_agent",
    shopping_cart_agent_tool_router,
    {
        "tools": "shopping_cart_agent_tool_node",
        "end": "coordinator_agent"
    }
)

workflow.add_conditional_edges(
    "warehouse_manager_agent",
    warehouse_manager_agent_tool_router,
    {
        "tools": "warehouse_manager_agent_tool_node",
        "end": "coordinator_agent"
    }
)

workflow.add_edge("product_qna_agent_tool_node", "product_qna_agent")
workflow.add_edge("shopping_cart_agent_tool_node", "shopping_cart_agent")
workflow.add_edge("warehouse_manager_agent_tool_node", "warehouse_manager_agent")


### Agent Execution

def agent_stream_wrapper(question: str, thread_id: str) -> dict:

    def _string_for_sse(string):
        return f"data: {string}\n\n"

    def _process_graph_event(chunk):

        def _is_node_start(chunk):
            return chunk[1].get("type") == "task"

        def _tool_to_text(tool_call):
            if tool_call.get("name") == "get_formatted_item_context":
                return f"Looking for items: {tool_call.get('args').get('query', '')}."
            elif tool_call.get("name") == "get_formatted_reviews_context":
                return f"Fetching user reviews..."
            elif tool_call.get("name") == "get_shopping_cart":
                    return f"Fetching shopping cart..."
            elif tool_call.get("name") == "remove_from_cart":
                return f"Removing items from shopping cart..."
            elif tool_call.get("name") == "add_to_shopping_cart":
                return f"Adding items to shopping cart..."
            elif tool_call.get("name") == "check_warehouse_availability":
                return f"Checking warehouse availability..."
            elif tool_call.get ("name") == "reserve_warehouse_items":
                return f"Reserving warehouse items..."

        if _is_node_start(chunk):
            if chunk[1].get("payload", {}).get("name") == "coordinator_agent":
                return "Analyzing the question..."
            if chunk[1].get("payload", {}).get("name") == "product_qna_agent":
                return "Planning..."
            if chunk[1].get("payload", {}).get("name") == "shopping_cart_agent":
                return "Planning..."
            if chunk[1].get("payload", {}).get("name") == "warehouse_manager_agent":
                return "Planning..."
            if chunk[1].get("payload", {}).get("name", "").endswith("tool_node"):
                message = " ".join([_tool_to_text(tool_call) for tool_call in chunk[1].get('payload', {}).get('input', {}).messages[-1].tool_calls])
                return message

    qdrant_client = QdrantClient(url="http://qdrant:6333")

    initial_state = {
        "messages": [HumanMessage(content=question)],
        "user_id": thread_id,
        "cart_id": thread_id,
        "coordinator_agent": {
            "iteration": 0,
            "final_answer": False,
            "plan": [],
            "next_agent": ""
        },
        "product_qna_agent": {
            "iteration": 0,
            "final_answer": False
        },
        "shopping_cart_agent": {
            "iteration": 0,
            "final_answer": False
        },
        "warehouse_manager_agent": {
            "iteration": 0,
            "final_answer": False
        }
    }
    config = {
            "configurable": {
                "thread_id": thread_id
            }
    }

    with PostgresSaver.from_conn_string(
            "postgresql://langgraph_user: langgraph_password@postgres:5432/langgraph_db"
    ) as checkpointer:

        graph = workflow.compile(
            checkpointer=checkpointer
        )
        
        for chunk in graph.stream(
            initial_state,
            config=config,
            stream_mode=["debug", "values"]
            ):

                processed_chunk = _process_graph_event(chunk)

                if processed_chunk:
                    yield _string_for_sse(processed_chunk)

                if chunk[0] == "values":
                    result = chunk[1]
    
    used_context = []
    
    for item in result.get("references", []):
        payload = qdrant_client.scroll(
            collection_name="Amazon-items-collection-01-hybrid-search",
            with_payload=True,
            with_vectors=False,
            scroll_filter=Filter(
                must=[
                    FieldCondition(
                        key="parent_asin",
                        match=MatchValue(value=item.get("id"))
                    )
                ]
            )
        )[0][0].payload
        image_url = payload.get("image", "")
        price = payload.get("price")
        if image_url:
            used_context.append(
                {
                    "image_url": image_url,
                    "price": price,
                    "description": item.get("description")
                }
            )

    shopping_cart = get_shopping_cart_for_sse(user_id=thread_id, cart_id=thread_id)

    shopping_cart_items = [
        {
            "price": float(item.get("price")) if item.get("price") else None,
            "quantity": item.get("quantity"),
            "currency": item.get("currency"),
            "product_image_url": item.get("product_image_url"),
            "total_price": float(item.get("total_price")) if item.get("total_price") else None
        }
        for item in shopping_cart
    ]
    
    yield _string_for_sse(json.dumps(
        {
            "type": "final_answer",
            "data": {
                "answer": result.get("answer", ""),
                "used_context": used_context,
                "trace_id": result.get("trace_id", ""),
                "shopping_cart": shopping_cart_items
            }
        }
    ))