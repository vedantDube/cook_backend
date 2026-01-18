import os
import warnings
from dotenv import load_dotenv
from langchain_google_genai import (
    GoogleGenerativeAIEmbeddings,
    ChatGoogleGenerativeAI
)
from langchain_pinecone import PineconeVectorStore
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate

warnings.filterwarnings("ignore")
load_dotenv()

chat_history = {}

# -----------------------------
# Initialization
# -----------------------------

def initialize_chatbot(user_index, collection_name):
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/text-embedding-004",
        google_api_key=os.getenv("GOOGLE_API_KEY")
    )

    vectorstore = PineconeVectorStore(
        index_name=user_index,
        embedding=embeddings,
        namespace=collection_name
    )

    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

    chat = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0,
        google_api_key=os.getenv("GOOGLE_API_KEY")
    )

    prompt = PromptTemplate(
        input_variables=["context", "chat_history", "question"],
        template="""
Answer the question based solely on the provided context stored in Pinecone.
Do not use any external knowledge.

If the context is empty or insufficient, respond with:
"I don’t have enough information to answer that based on the provided context."

Context:
{context}

Chat History:
{chat_history}

Question:
{question}

Answer:
"""
    )

    qa = ConversationalRetrievalChain.from_llm(
        llm=chat,
        retriever=retriever,
        combine_docs_chain_kwargs={"prompt": prompt},
        return_source_documents=True
    )

    return qa, vectorstore

# -----------------------------
# Reranking (NO embeddings)
# -----------------------------

def rerank_documents_with_scores(vectorstore, query_text, k=5):
    """
    Uses Pinecone similarity scores.
    No embedding calls.
    Frontend-compatible output.
    """
    results = vectorstore.similarity_search_with_score(query_text, k=k)

    reranked = sorted(results, key=lambda x: x[1], reverse=True)

    return [
        {
            "content": doc.page_content,
            "metadata": doc.metadata,
            "similarity": float(score)
        }
        for doc, score in reranked
    ]

# -----------------------------
# Main Query Function
# -----------------------------

def query_pinecone(query_text, user_index, collection_name):
    if user_index not in chat_history:
        chat_history[user_index] = {}

    if collection_name not in chat_history[user_index]:
        chat_history[user_index][collection_name] = []

    qa, vectorstore = initialize_chatbot(user_index, collection_name)

    # Retrieve + rerank using Pinecone scores
    reranked_documents = rerank_documents_with_scores(
        vectorstore,
        query_text,
        k=5
    )

    # Prepare context for LLM
    context = "\n\n".join(doc["content"] for doc in reranked_documents)

    res = qa({
        "question": query_text,
        "chat_history": chat_history[user_index][collection_name]
    })

    detailed_response = {
        "query": query_text,
        "retrieved_documents": [
            {
                "content": doc["content"],
                "metadata": doc["metadata"]
            }
            for doc in reranked_documents
        ],
        "reranked_documents": reranked_documents,
        "llm_response": res["answer"]
    }

    chat_history[user_index][collection_name].append(
        (query_text, res["answer"])
    )

    return detailed_response
