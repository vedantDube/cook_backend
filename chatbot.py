import os
import warnings
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.chains import ConversationalRetrievalChain
from langchain_pinecone import PineconeVectorStore
from langchain.prompts import PromptTemplate
from langchain.schema import Document
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

warnings.filterwarnings("ignore")
load_dotenv()

chat_history = {}

def initialize_chatbot(user_index, collection_name):
    """Initialize chatbot for a specific user's index."""
    embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004", google_api_key=os.getenv("GOOGLE_API_KEY"))
    
    vectorstore = PineconeVectorStore(
        index_name=user_index, embedding=embeddings, namespace=collection_name
    )
    
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})  # Retrieve top 5 documents

    chat = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        temperature=0,
        google_api_key=os.getenv("GOOGLE_API_KEY"),
        verbose=True
    )
    prompt_template = """
        Answer the question based solely on the provided context stored in Pinecone. Do not use any external knowledge or general information. If the context is empty or insufficient, respond with: "I don’t have enough information to answer that based on the provided context."

        Context: {context}
        Chat History: {chat_history}
        Question: {question}

        Answer:
        """
    prompt = PromptTemplate(input_variables=["context", "chat_history", "question"], template=prompt_template)
    qa = ConversationalRetrievalChain.from_llm(
        llm=chat,
        chain_type="stuff",
        retriever=retriever,
        combine_docs_chain_kwargs={"prompt": prompt},
        return_source_documents=True,
        verbose=True
    )
    return qa

def rerank_documents(query, documents, embeddings):
    """Re-rank retrieved documents based on cosine similarity to the query."""
    if not documents:
        return documents
    
    # Embed the query
    query_embedding = embeddings.embed_query(query)
    
    # Embed each document and calculate similarity
    reranked = []
    for doc in documents:
        doc_embedding = embeddings.embed_query(doc.page_content)
        similarity = cosine_similarity([query_embedding], [doc_embedding])[0][0]
        reranked.append((doc, similarity))
    
    # Sort by similarity (descending)
    reranked.sort(key=lambda x: x[1], reverse=True)
    return [doc for doc, _ in reranked]

def query_pinecone(query_text, user_index, collection_name):
    """Query chatbot with transparency: retrieved docs, re-ranked results, and final response."""
    if user_index not in chat_history:
        chat_history[user_index] = {}

    if collection_name not in chat_history[user_index]:
        chat_history[user_index][collection_name] = []

    qa = initialize_chatbot(user_index, collection_name)
    embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004", google_api_key=os.getenv("GOOGLE_API_KEY"))

    # Step 1: Retrieve raw documents
    vectorstore = PineconeVectorStore(index_name=user_index, embedding=embeddings, namespace=collection_name)
    raw_docs = vectorstore.similarity_search(query_text, k=5)

    # Step 2: Re-rank documents
    reranked_docs = rerank_documents(query_text, raw_docs, embeddings)

    # Step 3: Run the QA chain with re-ranked documents
    context = "\n\n".join([doc.page_content for doc in reranked_docs])
    res = qa({
        "question": query_text,
        "chat_history": chat_history[user_index][collection_name],
        "context": context  # Pass re-ranked context manually
    })

    # Prepare detailed response
    retrieved_info = [
        {
            "content": doc.page_content,
            "metadata": doc.metadata
        } for doc in raw_docs
    ]
    reranked_info = [
        {
            "content": doc.page_content,
            "metadata": doc.metadata,
            "similarity": cosine_similarity(
                [embeddings.embed_query(query_text)],
                [embeddings.embed_query(doc.page_content)]
            )[0][0]
        } for doc in reranked_docs
    ]

    detailed_response = {
        "query": query_text,
        "retrieved_documents": retrieved_info,
        "reranked_documents": reranked_info,
        "llm_response": res["answer"]
    }

    # Update chat history
    chat_history[user_index][collection_name].append((query_text, res["answer"]))

    # Log for debugging
    print("Query:", query_text)
    print("Retrieved Documents:", [doc.page_content for doc in raw_docs])
    print("Re-ranked Documents:", [doc.page_content for doc in reranked_docs])
    print("Final Answer:", res["answer"])

    return detailed_response