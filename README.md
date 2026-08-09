\# 🧠 InsightForge AI



\## Hybrid RAG Document Intelligence System



InsightForge AI is a local document question-answering system built using

Retrieval-Augmented Generation (RAG).



The system allows users to upload PDF documents and ask questions about

their content. It retrieves relevant information using Hybrid Retrieval,

combining Dense Vector Retrieval with BM25 Sparse Retrieval, and then uses

a local LLM to generate an answer based on the retrieved context.



\---



\# 1. Problem Statement



Traditional document search systems mainly depend on keyword matching.

This can fail when a user asks a question using different words from the

document.



InsightForge AI solves this problem by combining:



\- Semantic/Dense Retrieval

\- Keyword/BM25 Retrieval

\- Hybrid Retrieval

\- Local Large Language Model generation

\- Retrieval evaluation

\- Source attribution



This allows users to obtain answers that are grounded in their uploaded

documents.



\---



\# 2. Objectives



The main objectives of InsightForge AI are:



\- Extract text from PDF documents.

\- Divide documents into meaningful chunks.

\- Generate vector embeddings.

\- Store document embeddings in FAISS.

\- Perform semantic retrieval.

\- Perform keyword retrieval using BM25.

\- Combine both retrieval methods.

\- Generate answers using a local LLM.

\- Display the retrieved sources.

\- Display page and chunk information.

\- Evaluate retrieval quality.

\- Provide a simple user interface.



\---



\# 3. Technology Stack



| Component | Technology |

|---|---|

| Programming Language | Python |

| PDF Processing | pypdf |

| Text Chunking | Python |

| Embeddings | Local embedding model |

| Vector Database | FAISS |

| Sparse Retrieval | BM25 |

| Retrieval | Hybrid Retrieval |

| LLM | Llama 3.2:3b |

| LLM Runtime | Ollama |

| UI | Streamlit |

| Evaluation | Custom RAG Evaluator |



\---



\# 4. RAG Architecture



The overall pipeline is:



PDF Document

&#x20;    ↓

PDF Text Extraction

&#x20;    ↓

Text Chunking

&#x20;    ↓

Embedding Generation

&#x20;    ↓

FAISS Vector Store

&#x20;    ↓

&#x20;       ┌──────────────────────┐

&#x20;       │   User Question      │

&#x20;       └──────────┬───────────┘

&#x20;                  ↓

&#x20;       ┌──────────────────────┐

&#x20;       │ Hybrid Retrieval     │

&#x20;       │                      │

&#x20;       │ Dense + BM25         │

&#x20;       └──────────┬───────────┘

&#x20;                  ↓

&#x20;         Relevant Chunks

&#x20;                  ↓

&#x20;            Context Builder

&#x20;                  ↓

&#x20;          Llama 3.2:3b

&#x20;                  ↓

&#x20;             Final Answer

&#x20;                  ↓

&#x20;       Answer + Sources + Scores



\---



\# 5. PDF Processing



The PDF processing module extracts text from every page.



Each extracted page contains:



\- Page number

\- Document name

\- Page text



Example metadata:



```text

document\_name: UNIT 1 notes.pdf

page\_number: 1

