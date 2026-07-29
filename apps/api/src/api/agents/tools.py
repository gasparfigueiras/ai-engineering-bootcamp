import openai
import cohere
from qdrant_client import QdrantClient
from langsmith import traceable, get_current_run_tree
from qdrant_client.models import Prefetch, Document
from qdrant_client import models
from langchain_core.tools import tool


### Items metadata retrieval tool

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
def retrieve_data(query, qdrant_client, k=5, hybrid=True):

    query_embedding = get_embedding(query)

    if hybrid:
        results = qdrant_client.query_points(
            collection_name="Amazon-items-collection-01-hybrid-search",
            prefetch=[
                Prefetch(
                    query=query_embedding,
                    using="text-embedding-3-small",
                    limit=20
                ),
                Prefetch(
                    query=Document(
                        text=query,
                        model="qdrant/bm25"
                    ),
                    using="bm25",
                    limit=20
                )
            ],
            query=models.RrfQuery(rrf=models.Rrf(weights=[3,1])),
            limit=k
        )
    else:
        results = qdrant_client.query_points(
            collection_name="Amazon-items-collection-01-hybrid-search",
            query=query_embedding,
            using="text-embedding-3-small",
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
        name="rerank_data",
        run_type="tool"
)
def rerank_data(query, context, top_k=5):

    cohere_client = cohere.ClientV2()

    response = cohere_client.rerank(
        model="rerank-v4.0-pro",
        query=query,
        documents=context["retrieved_contexts"],
        top_n=top_k
    )

    order = [result.index for result in response.results]

    return {
        "retrieved_context_ids": [context["retrieved_context_ids"][i] for i in order],
        "retrieved_contexts": [context["retrieved_contexts"][i] for i in order],
        "similarity_scores": [context["similarity_scores"][i] for i in order],
        "retrieved_contexts_ratings": [context["retrieved_contexts_ratings"][i] for i in order]
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


@tool
def get_formatted_item_context(query: str, top_k: int = 5) -> str:

    """Search available products and return the top k matching inventory items.
    
    Expand the customer's question into 1-5 concise search statements and issue them
    in parallel in a single turn. Each statement covers one distinct product or
    attribute; no two may express the same intent. Use natural product-description
    language. If no brand or model is specified, search broadly rather than refusing.
    
        "Earphones for me and a waterproof speaker"
            -> "Personal earphones" | "Waterproof speaker"
        "A warm winter jacket for hiking"
            -> "Insulated winter jacket" | "Hiking outerwear for cold weather"
    
    Before calling, check what earlier calls in this conversation already returned.
    Search only for what is missing; results already retrieved remain valid and must
    not be fetched again.
    
    Args:
        query: A single search statement describing one product or attribute.
        top_k: Number of items to retrieve. Works best with 5 or more.
    
    Returns:
        A string of the top k available products, each prefixed with its ID and
        average rating.
        """

    qdrant_client = QdrantClient(url="http://qdrant:6333")

    retrieved_context = retrieve_data(
        query,
        qdrant_client,
        k=20
    )

    retrieved_context = rerank_data(query, retrieved_context, top_k=top_k)
    formatted_context = process_context(retrieved_context)

    return formatted_context
