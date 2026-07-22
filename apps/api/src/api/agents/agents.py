import instructor

from langsmith import traceable, get_current_run_tree
from pydantic import BaseModel, Field

from langchain_core.messages import SystemMessage, convert_to_openai_messages
from langchain_openai import ChatOpenAI

from api.agents.utils.prompt_management import prompt_template_config

from api.agents.tools import get_formatted_item_context


### QnA Agent Response Model

class RAGUsedContext(BaseModel):
    id: str = Field("ID of the item used to answer the question")
    description: str = Field(description="Description of the item used to answer the question")

class FinalResponse(BaseModel):
    answer: str = Field(description="Answer to the question")
    references: list[RAGUsedContext] = Field(description="List of items used to answer the question")


### Intent Router Response Model

class IntentRouterResponse(BaseModel):
    question_relevant: bool
    answer: str = Field(description="An answer to the question if the users question is not relevant to the products.")


### QnA Agent Node

@traceable(
    name="agent_node",
    run_type="llm",
    metadata={
        "ls_provider": "openai",
        "ls_model_name": "gpt-5.4-mini"
    }
)
def agent_node(state) -> dict:
    
    template = prompt_template_config("api/agents/prompts/qna_agent.yaml", "qna_agent")

    prompt = template.render()

    llm = ChatOpenAI(
        model="gpt-5.4-mini",
        reasoning_effort="none",
        use_responses_api=True
    )
    llm_with_tools = llm.bind_tools(
        [get_formatted_item_context, FinalResponse],
        tool_choice="any"
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

    if len(response.tool_calls) > 0:
        for tool_call in response.tool_calls:
            if tool_call.get("name") == "FinalResponse":
                final_answer = True
                answer = tool_call.get("args").get("answer")
                references.extend(tool_call.get("args").get("references"))

    return {
        "messages": [response],
        "final_answer": final_answer,
        "iteration": state.iteration + 1, 
        "answer": answer,
        "references": references
    }


### Intent Router Node

@traceable(
        name="route_intent",
        run_type="llm",
        metada={
            "ls_provider": "openai",
            "ls_model_name": "gpt-5.4-mini"
        }
)
def intent_router_node(state) -> dict:
    
    template = prompt_template_config("api/agents/prompts/intent_router_agent.yaml", "intent_router_agent")

    prompt = template.render()

    messages = state.messages

    conversation = []

    for message in messages:
        conversation.append(convert_to_openai_messages(message))

    client = instructor.from_provider(
        "openai/gpt-5.4-mini",
        mode=instructor.Mode.RESPONSES_TOOLS
    )

    response, raw_response = client.create_with_completion(
            messages=[
            {"role": "system", "content": prompt},
            *conversation
        ],
        reasoning={"effort": "none"},
        response_model=IntentRouterResponse
    )
    
    current_run = get_current_run_tree()
    if current_run:
        current_run.metadata["usage_metadata"] = {
            "input_tokens": raw_response.usage.input_tokens,
            "output_tokens": raw_response.usage.output_tokens,
            "total_tokens": raw_response.usage.total_tokens,
        }

    return {
        "question_relevant": response.question_relevant,
        "answer": response.answer
    }