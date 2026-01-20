# 🚀 COOKGPT — Multimodal RAG Chatbot  
### 🥇 Winner – 1st Place, IIIT Naya Raipur Hackathon Hack-o-Harbour 2025

COOKGPT is a **multimodal Retrieval-Augmented Generation (RAG) chatbot** built to ingest, index, and query information from **diverse data sources**—including documents, structured files, audio, and web links—while maintaining **retrieval transparency, explainability, and reliability**.

Developed as a solution to **Problem Statement 2 (AIML Track)** at the **IIIT Naya Raipur Hackathon**, COOKGPT successfully implemented **almost every deliverable** outlined in the challenge and secured **1st place**.

---

## 🧠 Problem Statement Overview

The goal was to build a chatbot capable of:

- Processing **heterogeneous data formats**:
  - PDFs (with images, links, nested links)
  - CSV files
  - Voice/audio files
  - Web URLs
- Converting extracted content into a **vector database**
- Supporting **automatic updates** when documents are modified
- Implementing **Retrieval-Augmented Generation (RAG)**
- Providing **transparent retrieval, re-ranking, and LLM responses**
- Allowing **multiple document collections**
- Exposing **APIs for programmatic access**
- Enforcing **guardrails** to reduce hallucinations and unsafe outputs

COOKGPT was designed to directly map these requirements into a **robust, end-to-end system**.

---

## ✨ Key Features

- 🔎 **Multimodal Data Ingestion**
  - PDFs (text, images, hyperlinks, nested links)
  - CSVs and structured data
  - Audio/voice inputs
  - Web pages and documentation

- 🧩 **End-to-End RAG Pipeline**
  - Vector-based retrieval
  - Context re-ranking
  - LLM-powered answer synthesis

- 🔄 **Automatic Vector DB Updates**
  - Detects document changes and refreshes embeddings automatically

- 🔍 **Retrieval Transparency & Explainability**
  - Displays retrieved chunks
  - Shows re-ranking results
  - Separates retrieved context from final LLM output

- 🗂️ **Multi-Collection Support**
  - Create, manage, and query independent document collections

- 🌐 **API Access**
  - Programmatic RAG-based querying for external applications

- 🛡️ **Guardrails & Safety**
  - Reduces hallucinations and ungrounded responses
  - Ensures secure document handling

---

## 🖼️ Application Screenshots

> _Add screenshots of the application UI, retrieval transparency, and querying interface here._

```text
📸 Screenshot 1 – Document Ingestion Interface
📸 Screenshot 2 – Retrieval & Re-ranking View
📸 Screenshot 3 – Chatbot Query Response

```
---

## 🧱 System Architecture

```text
┌──────────────┐
│   User Query │
└──────┬───────┘
       ↓
┌─────────────────────┐
│ Vector Retriever    │  ← Pinecone
└──────┬──────────────┘
       ↓
┌─────────────────────┐
│ Re-ranking Layer    │  ← Context Scoring
└──────┬──────────────┘
       ↓
┌─────────────────────┐
│ LLM Response Engine │  ← Gemini API
└──────┬──────────────┘
       ↓
┌────────────────────────────────────┐
│ Transparent Output                  │
│ Final LLM Response                  │
└────────────────────────────────────┘

