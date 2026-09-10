import instructor

from langsmith import traceable, get_current_run_tree
from pydantic import BaseModel, Field

from langchain_core.messages import SystemMessage, convert_to_openai_messages, AIMessage
from langchain_openai import ChatOpenAI

from api.agents.utils.prompt_management import prompt_template_config
from api.agents.tools import get_formatted_item_context, get_formatted_reviews_context, get_shopping_cart, remove_from_cart, add_to_shopping_cart, check_warehouse_availability, reserve_warehouse_items
from typing import List


### QnA Agent Response Model

class RAGUsedContext(BaseModel):
    id: str = Field("ID of the item used to answer the question")
    description: str = Field(description="Description of the item used to answer the question")

class FinalQnAAgentResponse(BaseModel):

    """Call this tool when the final answer is possible using available context."""

    answer: str = Field(description="Answer to the question")
    references: list[RAGUsedContext] = Field(description="List of items used to answer the question")

### General Agent Response Model

class FinalAgentResponse(BaseModel):

    answer: str = Field(description="Answer to the question")


### Coordinator Agent Response Model

class Delegation(BaseModel):
    agent: str = Field(description="The agent to delegate the task to.")
    task: str = Field(description="The task to be performed by the agent.")

class Plan(BaseModel):
    next_agent: str = Field(description="The next agent to invoke.")
    plan: List[Delegation] = Field(description="A list of delegations to agents with tasks to be performed in sequence.")



### QnA Agent Node

@traceable(
    name="product_qna_node",
    run_type="llm",
    metadata={
        "ls_provider": "openai",
        "ls_model_name": "gpt-5.4-mini"
    }
)
def product_qna_agent(state) -> dict:

    template = prompt_template_config("api/agents/prompts/qna_agent.yaml", "qna_agent")
    
    prompt = template.render()

    llm = ChatOpenAI(
        model="gpt-5.4-mini",
        reasoning_effort="low",
        use_responses_api=True
    )
    llm_with_tools = llm.bind_tools(
        [get_formatted_item_context, get_formatted_reviews_context, FinalQnAAgentResponse],
        tool_choice="required"
    )

    response = llm_with_tools.invoke(
        [
            SystemMessage(content=prompt),
            *state.messages
        ]
    )

    current_run = get_current_run_tree()
    if current_run:
        current_run.metadata["usage_metadata"] = {
            "input_tokens": response.usage_metadata["input_tokens"],
            "output_tokens": response.usage_metadata["output_tokens"],
            "total_tokens": response.usage_metadata["total_tokens"],
        }

    final_answer = False
    answer = ""
    references = []

    def sanitize_response(response):

        for tool_call in response.tool_calls:
            if tool_call.get("name") == "FinalQnAAgentResponse":
                answer = tool_call.get("args").get("answer")

        return AIMessage(content=answer)

    if len(response.tool_calls) > 0:
        for tool_call in response.tool_calls:
            if tool_call.get("name") == "FinalQnAAgentResponse":
                final_answer = True
                answer = tool_call.get("args").get("answer")
                references.extend(tool_call.get("args").get("references"))

                response = sanitize_response(response);

    return {
        "messages": [response],
        "product_qna_agent": {
            "iteration": state.product_qna_agent.iteration +1,
            "final_answer": final_answer
        },
        "answer": answer,
        "references": references
    }


### Shopping Cart Agent Node

@traceable(
    name="shopping_cart_agent",
    run_type="llm",
    metadata={
        "ls_provider": "openai",
        "ls_model_name": "gpt-5.4-mini"
    }
)
def shopping_cart_agent(state) -> dict:

    template = prompt_template_config("api/agents/prompts/shopping_cart_agent.yaml", "shopping_cart_agent")

    prompt = template.render(
        user_id=state.user_id,
        cart_id=state.cart_id
    )

    llm = ChatOpenAI(
        model="gpt-5.4-mini",
        reasoning_effort="low",
        use_responses_api=True
    )
    llm_with_tools = llm.bind_tools(
        [get_shopping_cart, remove_from_cart, add_to_shopping_cart, FinalAgentResponse],
        tool_choice="required"
    )

    response = llm_with_tools.invoke(
        [
            SystemMessage(content=prompt),
            *state.messages
        ]
    )

    current_run = get_current_run_tree()
    if current_run:
            current_run.metadata["usage_metadata"] = {
                "input_tokens": response.usage_metadata["input_tokens"],
                "output_tokens": response.usage_metadata["output_tokens"],
                "total_tokens": response.usage_metadata["total_tokens"],
            }

    final_answer = False
    answer = ""

    def sanitize_response(response):

        for tool_call in response.tool_calls:
            if tool_call.get("name") == "FinalAgentResponse":
                answer = tool_call.get("args").get("answer")

        return AIMessage(content=answer)

    if len(response.tool_calls) > 0:
        for tool_call in response.tool_calls:
            if tool_call.get("name") == "FinalAgentResponse":
                final_answer = True
                answer = tool_call.get("args").get("answer")

                response = sanitize_response(response);

    return {
        "messages": [response],
        "shopping_cart_agent": {
            "iteration": state.shopping_cart_agent.iteration +1,
            "final_answer": final_answer
        },
        "answer": answer,
    }


### Warehouse Manager Agent Node

@traceable(
    name="warehouse_manager_agent",
    run_type="llm",
    metadata={
        "ls_provider": "openai",
        "ls_model_name": "gpt-5.4-mini"
    }
)
def warehouse_manager_agent(state) -> dict:

    template = prompt_template_config("api/agents/prompts/warehouse_manager_agent.yaml", "warehouse_manager_agent")

    prompt = template.render()

    llm = ChatOpenAI(
        model="gpt-5.4-mini",
        reasoning_effort="low",
        use_responses_api=True
    )
    llm_with_tools = llm.bind_tools(
        [check_warehouse_availability, reserve_warehouse_items, FinalAgentResponse],
        tool_choice="required"
    )

    response = llm_with_tools.invoke(
        [
            SystemMessage(content=prompt),
            *state.messages
        ]
    )

    current_run = get_current_run_tree()
    if current_run:
            current_run.metadata["usage_metadata"] = {
                "input_tokens": response.usage_metadata["input_tokens"],
                "output_tokens": response.usage_metadata["output_tokens"],
                "total_tokens": response.usage_metadata["total_tokens"],
            }

    final_answer = False
    answer = ""

    def sanitize_response(response):

        for tool_call in response.tool_calls:
            if tool_call.get("name") == "FinalAgentResponse":
                answer = tool_call.get("args").get("answer")

        return AIMessage(content=answer)

    if len(response.tool_calls) > 0:
        for tool_call in response.tool_calls:
            if tool_call.get("name") == "FinalAgentResponse":
                final_answer = True
                answer = tool_call.get("args").get("answer")

                response = sanitize_response(response);

    return {
        "messages": [response],
        "warehouse_manager_agent": {
            "iteration": state.warehouse_manager_agent.iteration +1,
            "final_answer": final_answer
        },
        "answer": answer,
    }


### Coordinator Agent Node

@traceable(
    name="coordinator_agent",
    run_type="llm",
    metadata={
        "ls_provider": "openai",
        "ls_model_name": "gpt-5.4-mini"
    }
)
def coordinator_agent(state) -> dict:

    template = prompt_template_config("api/agents/prompts/coordinator_agent.yaml", "coordinator_agent")
    
    prompt = template.render()

    llm = ChatOpenAI(
        model="gpt-5.4-mini",
        reasoning_effort="low",
        use_responses_api=True
    )
    llm_with_tools = llm.bind_tools(
        [FinalAgentResponse, Plan],
        tool_choice="required"
    )

    response = llm_with_tools.invoke(
        [
            SystemMessage(content=prompt),
            *state.messages
        ]
    )

    current_run = get_current_run_tree()
    if current_run:
        current_run.metadata["usage_metadata"] = {
            "input_tokens": response.usage_metadata["input_tokens"],
            "output_tokens": response.usage_metadata["output_tokens"],
            "total_tokens": response.usage_metadata["total_tokens"],
        }
        trace_id = str(current_run.trace_id)
    else:
        trace_id = ""
        
    final_answer = False
    answer = ""
    plan = []
    next_agent = ""

    def sanitize_response(response):

        for tool_call in response.tool_calls:
            if tool_call.get("name") == "FinalAgentResponse":
                answer = tool_call.get("args").get("answer")

        return AIMessage(content=answer)

    if len(response.tool_calls) > 0:
        if (response.tool_calls[0].get("name") == "Plan"):
            plan = response.tool_calls[0].get("args").get("plan")
            next_agent = response.tool_calls[0].get("args").get("next_agent")
            response = None
        else:
            for tool_call in response.tool_calls:
                if tool_call.get("name") == "FinalAgentResponse":
                    final_answer = True
                    answer = tool_call.get("args").get("answer")

                    response = sanitize_response(response);

    return {
        "messages": [response] if response else [],
        "coordinator_agent": {
            "iteration": state.coordinator_agent.iteration +1,
            "final_answer": final_answer,
            "plan": plan,
            "next_agent": next_agent
        },
        "answer": answer,
        "trace_id": trace_id
    }