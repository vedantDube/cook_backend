import os
import warnings
from dotenv import load_dotenv
from langchain_google_genai import (
    GoogleGenerativeAIEmbeddings,
    ChatGoogleGenerativeAI
)
from langchain_pinecone import PineconeVectorStore
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from pydantic import SecretStr

warnings.filterwarnings("ignore")
load_dotenv()

chat_history = {}

# -----------------------------
# Setup
# -----------------------------

def initialize_llm():
    google_api_key = os.getenv("GOOGLE_API_KEY")
    return ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0,
        google_api_key=(google_api_key.get_secret_value() if isinstance(google_api_key, SecretStr) else google_api_key)
    )

def initialize_vectorstore(user_index, collection_name):
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/text-embedding-gecko-001",
        google_api_key=os.getenv("GOOGLE_API_KEY")
    )

    return PineconeVectorStore(
        index_name=user_index,
        embedding=embeddings,
        namespace=collection_name
    )

# -----------------------------
# Query
# -----------------------------

def query_pinecone(query_text, user_index, collection_name):
    if user_index not in chat_history:
        chat_history[user_index] = {}

    if collection_name not in chat_history[user_index]:
        chat_history[user_index][collection_name] = []

    vectorstore = initialize_vectorstore(user_index, collection_name)

    # 🔹 ONE embedding call happens here (unavoidable & correct)
    results = vectorstore.similarity_search_with_score(query_text, k=5)

    reranked_documents = [
        {
            "content": doc.page_content,
            "metadata": doc.metadata,
            "similarity": float(score)
        }
        for doc, score in results
    ]

    context = "\n\n".join(d["content"] for d in reranked_documents)

    llm = initialize_llm()

    prompt = PromptTemplate(
        input_variables=["context", "chat_history", "question"],
        template="""
Answer the question based ONLY on the provided context.
If the context is insufficient, say:
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

    chain = LLMChain(llm=llm, prompt=prompt)

    answer = chain.run(
        context=context,
        chat_history=chat_history[user_index][collection_name],
        question=query_text
    )

    chat_history[user_index][collection_name].append(
        (query_text, answer)
    )

    return {
        "query": query_text,
        "retrieved_documents": [
            {"content": d["content"], "metadata": d["metadata"]}
            for d in reranked_documents
        ],
        "reranked_documents": reranked_documents,
        "llm_response": answer
    }
