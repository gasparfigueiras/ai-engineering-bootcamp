import openai
import cohere
from qdrant_client import QdrantClient
from langsmith import traceable, get_current_run_tree
from qdrant_client.models import Prefetch, Document, FusionQuery, Filter, FieldCondition, MatchAny, MatchValue
from qdrant_client import models
from langchain_core.tools import tool

import psycopg2
from psycopg2.extras import RealDictCursor
import numpy as np


@traceable(
        name="embed_query",
        run_type="embedding",
        metadata={
            "ls_provider":"openai",
            "ls_model_name": "text_embedding-2-small"
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

### Items metadata retrieval tool

@traceable(
        name="retrieve_data",
        run_type="retriever"
)
def retrieve_items_data(query, qdrant_client, k=5, hybrid=True):

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
        name="format_retrieved_context",
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

    qdrant_client = QdrantClient(url="http://localhost:6333")

    retrieved_context = retrieve_items_data(
        query,
        qdrant_client,
        k=20
    )

    retrieved_context = rerank_data(query, retrieved_context, top_k=top_k)
    formatted_context = process_context(retrieved_context)

    return formatted_context


### Reviews metadata retrieval tool

@traceable(
        name="retrieve_prefiltered_reviews_data",
        run_type="retriever"
)
def retrieve_prefiltered_reviews_data(query, parent_asins, qdrant_client, k=5):

    query_embedding = get_embedding(query)

    results = qdrant_client.query_points(
        collection_name="Amazon-reviews-collection-01",
        prefetch=[
            Prefetch(
                query=query_embedding,
                using="text-embedding-3-small",
                filter=Filter(
                    must=[
                        FieldCondition(
                            key="parent_asin",
                            match=MatchAny(
                                any=parent_asins
                            )
                        )
                    ]
                ),
                limit=20
            )
        ],
         query=FusionQuery(fusion="rrf"),
        limit=k
    )

    retrieved_context_ids = []
    retrieved_contexts = []
    similarity_scores = []


    for result in results.points:
        retrieved_context_ids.append(result.payload["parent_asin"])
        retrieved_contexts.append(result.payload["preprocessed_data"])
        similarity_scores.append(result.score)

    return {
        "retrieved_context_ids": retrieved_context_ids,
        "retrieved_contexts": retrieved_contexts,
        "similarity_scores": similarity_scores,
    }

@traceable(
        name="format_retrieved_context",
        run_type="prompt"
)
def process_context_reviews(context):
    
    formated_context = ""

    for id, chunk in zip(context["retrieved_context_ids"], context["retrieved_contexts"]):
        formated_context += f"- ID: {id}, user review: {chunk}\n"

    return formated_context

@tool
def get_formatted_reviews_context(query: str, parent_asins: list[str], top_k: int = 5) -> str:

    """Get the top k reviews matching a query for a list of prefiltered items.

    Args:
        query: The query to get the top k reviews for
        item_list: The list of item IDs to prefilter for before running the query
        top_k: The number of reviews to retrieve, this should be at least 20 if multipple items are prefiltered

    Returns:
        A string of the top k context chunks with IDs prepending each chunk, each representing a review for a given inventory item for a given query.
    """

    qdrant_client = QdrantClient(url="http://localhost:6333")

    retrieved_context = retrieve_prefiltered_reviews_data(
        query,
        parent_asins,
        qdrant_client,
        k=20
    )
    formatted_context = process_context_reviews(retrieved_context)

    return formatted_context


### Shopping Cart Tools ###

# Add shopping Cart Tool

@tool
def add_to_shopping_cart(items: list[dict], user_id: str, cart_id: str) -> str:

    """Add a list of provided items to the shopping cart.
    
    Args:
        items: A list of items to add to the shopping cart. Each item is a dictionary with the following keys: product_id, quantity.
        user_id: The id of the user to add the items to the shopping cart.
        cart_id: The id of the shopping cart to add the items to.
        
    Returns:
        A list of the items added to the shopping cart.
    """

    conn = psycopg2.connect(
        host="localhost",
        port=5433,
        database="tools_database",
        user="tools_user",
        password="tools_user_password"
    )
    conn.autocommit = True

    with conn.cursor(cursor_factory=RealDictCursor) as cursor:
        
        for item in items:
            product_id = item['product_id']
            quantity = item['quantity']

            qdrant_client = QdrantClient(url="http://localhost:6333")

            dummy_vector = np.zeros(1536).tolist()
            payload = qdrant_client.query_points(
                collection_name="Amazon-items-collection-01-hybrid-search",
                prefetch=[
                    Prefetch(
                        query=dummy_vector,
                        filter=Filter(
                            must=[
                                FieldCondition(
                                    key="parent_asin",
                                    match=MatchValue(value=product_id)
                                )
                            ]
                    ),
                        using="text-embedding-3-small",
                        limit=20
                    )
                ],
                query=FusionQuery(fusion="rrf"),
                limit=1,
            ).points[0].payload

            product_image_url = payload.get("image")
            price = payload.get("price")
            currency = 'USD'
        
            # Check if item already exists
            check_query = """
                SELECT id, quantity, price 
                FROM shopping_carts.shopping_cart_items 
                WHERE user_id = %s AND shopping_cart_id = %s AND product_id = %s
            """
            cursor.execute(check_query, (user_id, cart_id, product_id))
            existing_item = cursor.fetchone()
            
            if existing_item:
                # Update existing item
                new_quantity = existing_item['quantity'] + quantity
                
                update_query = """
                    UPDATE shopping_carts.shopping_cart_items 
                    SET 
                        quantity = %s,
                        price = %s,
                        currency = %s,
                        product_image_url = COALESCE(%s, product_image_url)
                    WHERE user_id = %s AND shopping_cart_id = %s AND product_id = %s
                    RETURNING id, quantity, price
                """
                
                cursor.execute(update_query, (new_quantity, price, currency, product_image_url, user_id, cart_id, product_id))
            
            else:
                # Insert new item
                insert_query = """
                    INSERT INTO shopping_carts.shopping_cart_items (
                        user_id, shopping_cart_id, product_id,
                        price, quantity, currency, product_image_url
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s)
                    RETURNING id, quantity, price
                """
                
                cursor.execute(insert_query, (user_id, cart_id, product_id, price, quantity, currency, product_image_url))
            
    return f"Added {items} to the shopping cart."


# Get Shopping Cart Tool

@tool
def get_shopping_cart(user_id: str, cart_id: str) -> list[dict]:

    """
    Retrieve all items in a user's shopping cart.
    
    Args:
        user_id: User identifier
        cart_id: Cart identifier
    
    Returns:
        List of dictionaries containing cart items
    """
    
    conn = psycopg2.connect(
        host="localhost",
        port=5433,
        database="tools_database",
        user="tools_user",
        password="tools_user_password"
    )
    conn.autocommit = True

    with conn.cursor(cursor_factory=RealDictCursor) as cursor:

        query = """
                SELECT 
                    product_id, price, quantity,
                    currency, product_image_url,
                    (price * quantity) as total_price
                FROM shopping_carts.shopping_cart_items 
                WHERE user_id = %s AND shopping_cart_id = %s
                ORDER BY added_at DESC
            """
        cursor.execute(query, (user_id, cart_id))

        return [dict(row) for row in cursor.fetchall()]


# Remove from Shopping Cart Tool

@tool
def remove_from_cart(product_id: str, user_id: str, cart_id: str) -> str:

    """
    Remove an item completely from the shopping cart.
    
    Args:
        product_id: Product identifier to remove
        user_id: User identifier
        cart_id: Cart identifier
    
    Returns:
        Information about the removal of the item from the shopping cart.
    """
    
    conn = psycopg2.connect(
        host="localhost",
        port=5433,
        database="tools_database",
        user="tools_user",
        password="tools_user_password"
    )
    conn.autocommit = True

    with conn.cursor(cursor_factory=RealDictCursor) as cursor:

        query = """
                DELETE FROM shopping_carts.shopping_cart_items
                WHERE user_id = %s AND shopping_cart_id = %s AND product_id = %s
            """
        cursor.execute(query, (user_id, cart_id, product_id))

        return f"Removed {product_id} from the shopping cart." if cursor.rowcount > 0 else f"Item {product_id} not found in the shopping cart."