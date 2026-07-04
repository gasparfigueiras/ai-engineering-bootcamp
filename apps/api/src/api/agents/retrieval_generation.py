import openai
from qdrant_client import QdrantClient
from langsmith import traceable, get_current_run_tree
import instructor
from pydantic import BaseModel, Field
from qdrant_client.models import Filter, FieldCondition, MatchValue

class RAGUsedContext(BaseModel):
    id: str = Field("ID of the item used to answer the question")
    description: str = Field(description="Description of the item used to answer the question")

class RAGGenerationResponse(BaseModel):
    answer: str = Field(description="Answer to the question")
    references: list[RAGUsedContext] = Field(description="List of items used to answer the question")

@traceable(
        name="embed_query",
        run_type="embedding",
        metadata={
            "ls_provider": "openai",
            "ls_model_name": "text-embedding-3-small"
        }
)
def get_embedding(text, model="text-embedding-3-small"):
    response = openai.embeddings.create(
        input=text,
        model=model
    )

    current_run = get_current_run_tree()
    if current_run:
        current_run.metadata["usage_metadata"] = {
            "input_tokens": response.usage.prompt_tokens,
            "total_tokens": response.usage.total_tokens,
        }

    return response.data[0].embedding


@traceable(
        name="retrieve_data",
        run_type="retriever"
)
def retrieve_data(query, qdrant_client, k=5):

    query_embbeding = get_embedding(query)

    results = qdrant_client.query_points(
        collection_name="Amazon-items-collection-01",
        query=query_embbeding,
        limit=k
    )

    retrieved_context_ids = []
    retrieved_contexts = []
    similarity_scores = []
    retrieved_contexts_ratings = []

    for result in results.points:
        retrieved_context_ids.append(result.payload["parent_asin"])
        retrieved_contexts.append(result.payload["preprocessed_description"])
        similarity_scores.append(result.score)
        retrieved_contexts_ratings.append(result.payload["average_rating"])

    return {
        "retrieved_context_ids": retrieved_context_ids,
        "retrieved_contexts": retrieved_contexts,
        "similarity_scores": similarity_scores,
        "retrieved_contexts_ratings": retrieved_contexts_ratings
    }


@traceable(
        name="format_retrieved_conext",
        run_type="prompt"
)
def process_context(context):
    
    formated_context = ""

    for id, chunk, rating in zip(context["retrieved_context_ids"], context["retrieved_contexts"], context["retrieved_contexts_ratings"]):
        formated_context += f"- ID: {id},  rating: {rating},  description: {chunk}\n"

    return formated_context


@traceable(
        name="build_prompt",
        run_type="prompt"
)
def build_prompt(preprocessed_context, question):
    prompt = f"""

You are a shopping assistant that can answer questions about the products in stock.

You will be given a question and a list of context.

Instructions:
- Answer the question based on the provided context only.
- Never user word context ad refer to ir as the available products.
- Do not use markdown formatting.

Context: 
{preprocessed_context}

Question:
{question}
"""
    
    return prompt


@traceable(
        name="generate_answer",
        run_type="llm",
        metadata={
            "ls_provider": "openai",
            "ls_model_name": "gpt-5.4-nano"
        }
)
def generate_answer(prompt):

    client = instructor.from_provider(
    "openai/gpt-5.4-nano",
    mode=instructor.Mode.RESPONSES_TOOLS
    )

    response, raw_response = client.create_with_completion(
        messages=[
            {"role": "system", "content": prompt},
        ],
        reasoning={"effort": "none"},
        response_model=RAGGenerationResponse
    )

    current_run = get_current_run_tree()
    if current_run:
        current_run.metadata["usage_metadata"] = {
            "input_tokens": raw_response.usage.input_tokens,
            "output_tokens": raw_response.usage.output_tokens,
            "total_tokens": raw_response.usage.total_tokens,
        }

    return response


@traceable(
        name="rag_pipeline"
)
def rag_pipeline(question, qdrant_client, top_k=5):

    retrieved_context = retrieve_data(question, qdrant_client, k=top_k)
    preprocessed_context = process_context(retrieved_context)
    prompt = build_prompt(preprocessed_context, question)
    answer = generate_answer(prompt)

    final_answer = {
        "answer": answer.answer,
        "references": answer.references,
        "question": question,
        "retrieved_context_ids": retrieved_context["retrieved_context_ids"],
        "retrieved_context": retrieved_context["retrieved_contexts"]
    }

    return final_answer

def rag_pipeline_wraper(question, top_k=5):

    qdrant_client = QdrantClient(url="http://qdrant:6333")

    result = rag_pipeline(question, qdrant_client, top_k)

    used_context = []

    for item in result.get("references", []):
        payload = qdrant_client.scroll(
        collection_name="Amazon-items-collection-01",
        with_payload=True,
        with_vectors=False,
        scroll_filter=Filter(
            must=[
                FieldCondition(
                    key="parent_asin",
                    match=MatchValue(value=item.id)
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
                    "description": item.description
                }
            )

    return {
        "answer": result["answer"],
        "used_context": used_context
    }