import openai
from qdrant_client import QdrantClient
from langsmith import traceable, get_current_run_tree

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

    response = openai.chat.completions.create(
        model="gpt-5.4-nano",
        messages=[
            {"role": "system", "content": prompt},
        ],
        reasoning_effort="none"
    )

    current_run = get_current_run_tree()
    if current_run:
        current_run.metadata["usage_metadata"] = {
            "input_tokens": response.usage.prompt_tokens,
            "output_tokens": response.usage.completion_tokens,
            "total_tokens": response.usage.total_tokens,
        }

    return response.choices[0].message.content


@traceable(
        name="rag_pipeline"
)
def rag_pipeline(question, qdrant_client, top_k=5):

    retrieved_context = retrieve_data(question, qdrant_client, k=top_k)
    preprocessed_context = process_context(retrieved_context)
    prompt = build_prompt(preprocessed_context, question)
    answer = generate_answer(prompt)

    final_answer = {
        "answer": answer,
        "question": question,
        "retrieved_context_ids": retrieved_context["retrieved_context_ids"],
        "retrieved_context": retrieved_context["retrieved_contexts"]
    }

    return final_answer