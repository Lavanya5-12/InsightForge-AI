# 🧠 InsightForge AI

## Hybrid RAG Document Intelligence System

InsightForge AI is a **local Hybrid Retrieval-Augmented Generation (RAG) based document question-answering system** that allows users to upload one or more PDF documents and ask natural-language questions about their content.

The system combines **Dense Vector Retrieval using FAISS** with **Sparse Keyword Retrieval using BM25**. The retrieved information is then provided as context to a locally running **Llama 3.2:3b** model through **Ollama** to generate a document-grounded response.

The application is developed using **Python and Streamlit** and provides document statistics, source attribution, page/chunk information, retrieval scores, and a conversational question-answering interface.

---
⚡ Quick Start

1. Clone the repository

git clone https://github.com/Lavanya5-12/InsightForge-AI.git
cd InsightForge-AI

2. Create and activate a virtual environment
Windows:
python -m venv venv
venv\Scripts\activate

3. Install dependencies
pip install -r requirements.txt

4. Install the required Ollama model
Make sure Ollama is installed, then run:

ollama pull llama3.2:3b

Verify:

ollama list

5. Start the application
streamlit run app.py

Open the application in your browser:

http://localhost:8501

Prerequisites: Python 3.10+, Git, Ollama, and sufficient system resources to run the local embedding model and Llama 3.2:3b.

For complete installation instructions, architecture, project structure, usage guide, troubleshooting, limitations, and future enhancements, continue reading below.




## 📌 Project Overview

Traditional document search systems often depend primarily on keyword matching. This can fail when the user's question uses different wording from the document.

InsightForge AI addresses this limitation by combining two complementary retrieval approaches:

* 🔎 **Dense/Semantic Retrieval** using FAISS
* 🔤 **Sparse/Keyword Retrieval** using BM25
* 🔀 **Hybrid Retrieval** combining both approaches
* 🧠 **Local LLM Generation** using Llama 3.2:3b
* 📚 **Source Attribution** with document, page, and chunk information
* 📊 **Retrieval Scores and System Analytics**
* 💬 **Interactive conversational interface**

The goal is to provide answers that are grounded in the content of the uploaded documents rather than relying only on the pretrained knowledge of the LLM.

---

# 🎯 Problem Statement

When users have large PDF documents, manually searching through them to find specific information can be time-consuming.

A simple keyword-based search system has another limitation: it may fail when the user's wording differs from the exact terminology used in the document.

For example:

**Document:**

> "The automobile industry experienced significant growth."

**User question:**

> "How did the car industry develop?"

A pure keyword search may not identify the relevant information because the exact word "car" does not appear.

InsightForge AI combines semantic retrieval and keyword retrieval to address both types of search requirements.

---

# 🎯 Objectives

The main objectives of InsightForge AI are:

* Extract text from PDF documents.
* Support multiple PDF uploads.
* Preserve document and page-level metadata.
* Divide extracted content into meaningful chunks.
* Generate vector embeddings for document chunks.
* Store embeddings in a FAISS index.
* Perform dense semantic retrieval.
* Perform keyword-based retrieval using BM25.
* Combine dense and sparse retrieval results.
* Use weighted hybrid retrieval.
* Construct relevant context for the LLM.
* Generate answers using a locally running Llama model.
* Display document, page, and chunk information.
* Display retrieval scores.
* Maintain conversational chat history.
* Provide document statistics.
* Provide a simple interactive Streamlit interface.

---

# 🏗️ System Architecture

## Overall Architecture

```text
                    PDF DOCUMENTS
                         │
                         ▼
                ┌─────────────────┐
                │ PDF Text        │
                │ Extraction      │
                └────────┬────────┘
                         │
                         ▼
                ┌─────────────────┐
                │ Text Chunking   │
                └────────┬────────┘
                         │
                         ▼
                ┌─────────────────┐
                │ Embedding       │
                │ Generation      │
                └────────┬────────┘
                         │
                         ▼
                ┌─────────────────┐
                │ FAISS Vector    │
                │ Index           │
                └─────────────────┘


                  USER QUESTION
                         │
             ┌───────────┴───────────┐
             ▼                       ▼
      Dense Retrieval          BM25 Retrieval
          FAISS                    BM25
             │                       │
             └───────────┬───────────┘
                         ▼
                Hybrid Retrieval
                         │
                  60% Dense
                  40% BM25
                         │
                         ▼
                 Relevant Chunks
                         │
                         ▼
                 Context Builder
                         │
                         ▼
                Llama 3.2:3b
                   via Ollama
                         │
                         ▼
                   Final Answer
                         │
                         ▼
             Sources + Retrieval Scores
```

---

# 🔄 Complete Workflow

## 1. Document Upload

The user can upload **one or more PDF documents** through the Streamlit sidebar.

The uploaded files are temporarily stored under:

```text
data/uploads/
```

The application keeps track of:

* Document names
* Number of pages
* Number of generated chunks

---

## 2. PDF Text Extraction

The PDF processing module extracts text from each page.

For every page, metadata is maintained, including:

```text
document_name
page_number
page_text
```

This metadata is important because it allows the system to later identify where retrieved information originated.

---

## 3. Text Chunking

The extracted document content is divided into smaller chunks.

Chunking is necessary because sending an entire large document to an LLM is inefficient and may exceed the model's context capacity.

Chunking helps:

* Improve retrieval precision
* Reduce unnecessary context
* Make documents searchable at a smaller granularity
* Provide more focused information to the LLM

Each chunk retains relevant metadata such as its document and page information.

---

# 🧠 Embedding Generation

InsightForge AI uses:

### Embedding Model

**mxbai-embed-large**

The embedding model converts each text chunk into a numerical vector representation.

Conceptually:

```text
Text Chunk
     │
     ▼
mxbai-embed-large
     │
     ▼
Numerical Vector
     │
     ▼
FAISS Index
```

During querying, the user's question is also converted into an embedding so that semantically similar document chunks can be retrieved.

---

# 🔎 Dense Retrieval with FAISS

FAISS is used for efficient similarity search over the generated embedding vectors.

The dense retrieval process is:

```text
User Question
      │
      ▼
Question Embedding
      │
      ▼
FAISS Similarity Search
      │
      ▼
Semantically Relevant Chunks
```

Dense retrieval is useful when the user's question and the document use different words but have similar meanings.

### Example

```text
Question:
"How can memory consumption be reduced?"

Document:
"Techniques for minimizing RAM usage..."
```

Semantic retrieval can identify the relationship even though the exact wording differs.

---

# 🔤 Sparse Retrieval with BM25

BM25 is used for keyword-based retrieval.

It ranks document chunks based on the relevance of terms appearing in the user's query.

BM25 is particularly useful for:

* Exact technical terms
* Names
* Abbreviations
* Specific terminology
* Keywords
* Identifiers

The BM25 process is:

```text
User Question
      │
      ▼
Keyword Matching
      │
      ▼
BM25 Ranking
      │
      ▼
Relevant Chunks
```

---

# 🔀 Hybrid Retrieval

The key feature of InsightForge AI is **Hybrid Retrieval**.

Instead of relying only on semantic retrieval or only on keyword retrieval, the system combines both.

The current application configuration uses:

```text
Dense Retrieval = 60%
BM25 Retrieval  = 40%
```

Conceptually:

```text
                User Question
                      │
          ┌───────────┴───────────┐
          ▼                       ▼
     FAISS Search             BM25 Search
       60%                       40%
          │                       │
          └───────────┬───────────┘
                      ▼
               Hybrid Ranking
                      │
                      ▼
              Relevant Chunks
```

### Why use Hybrid Retrieval?

Dense retrieval is strong at understanding **semantic meaning**, while BM25 is strong at identifying **exact terms and keywords**.

Combining them makes retrieval more robust.

### Dense Retrieval

**Strength:** Understands semantic similarity.

**Limitation:** Exact terminology may not always receive the highest ranking.

### BM25

**Strength:** Strong lexical/keyword matching.

**Limitation:** Can struggle when the user's wording differs significantly from the document.

### Hybrid Retrieval

Combines the strengths of both.

---

# 📚 Retrieval-Augmented Generation

After the relevant chunks are retrieved, the system constructs a context from those chunks.

The context is then supplied to the local LLM.

The complete RAG process is:

```text
User Question
      │
      ▼
Hybrid Retrieval
      │
      ▼
Relevant Document Chunks
      │
      ▼
Context Construction
      │
      ▼
Llama 3.2:3b
      │
      ▼
Generated Answer
```

The LLM therefore receives information retrieved from the user's documents before generating the response.

---

# 🤖 Local LLM — Llama 3.2:3b

InsightForge AI uses:

### Model

**Llama 3.2:3b**

### Runtime

**Ollama**

Ollama is used to run the LLM locally.

Using a local model provides:

* Local inference
* Reduced dependence on external LLM APIs
* Greater control over document processing
* No per-request external LLM API requirement

The 3B model also makes the project more practical for a local development environment.

However, smaller local models can have lower reasoning and generation capabilities than larger hosted models.

---

# 🖥️ Streamlit Application

The user interface is built using **Streamlit**.

The application provides:

### 📄 Document Management

* Upload one or more PDF files
* Display loaded documents
* Display document count
* Display page count
* Display chunk count

### 💬 Conversational Interface

* Ask questions about uploaded documents
* Maintain chat history
* Display previous questions and answers
* Clear chat history

### 🔬 System Analytics

The sidebar displays:

* Retrieval method
* Generation model
* Embedding model
* Retrieval configuration

Current configuration:

```text
Retrieval:
FAISS + BM25

Weights:
Dense 60% + BM25 40%

Embedding Model:
mxbai-embed-large

LLM:
Llama 3.2:3b
```

---

# 📚 Source Attribution

InsightForge AI displays source information along with generated responses.

For retrieved chunks, the application can display information such as:

* Document name
* Page number
* Chunk number
* Dense retrieval score
* BM25 score
* Hybrid/retrieval information

This improves transparency and makes it easier for users to inspect the retrieved evidence behind an answer.

---

# 📊 Retrieval Evaluation

A RAG system should not be evaluated only by looking at whether the final answer sounds correct.

The retrieval stage also needs to be evaluated.

InsightForge AI includes an evaluation component for assessing retrieval behavior.

Common retrieval metrics include:

### Precision@K

Measures the proportion of retrieved top-K results that are relevant.

### Recall@K

Measures how many of the relevant results were successfully retrieved.

### Hit Rate

Measures whether at least one relevant result appears in the retrieved results.

### Mean Reciprocal Rank (MRR)

Measures how highly the first relevant result appears in the ranking.

Evaluation can help identify whether an incorrect answer is caused by:

```text
Poor Document Processing
          OR
Poor Retrieval
          OR
Poor Context
          OR
LLM Generation
```

---

# 🛠️ Technology Stack

| Component            | Technology            |
| -------------------- | --------------------- |
| Programming Language | Python                |
| User Interface       | Streamlit             |
| PDF Processing       | pypdf                 |
| Embedding Model      | mxbai-embed-large     |
| Dense Retrieval      | FAISS                 |
| Sparse Retrieval     | BM25                  |
| Retrieval Strategy   | Hybrid Retrieval      |
| Dense/BM25 Weights   | 60% / 40%             |
| LLM                  | Llama 3.2:3b          |
| LLM Runtime          | Ollama                |
| Evaluation           | Custom RAG Evaluation |
| Version Control      | Git / GitHub          |

---

# 📁 Project Structure

```text
InsightForge-AI/
│
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
│
├── src/
│   ├── __init__.py
│   ├── chunker.py
│   ├── embeddings.py
│   ├── evaluator.py
│   ├── hybrid_retriever.py
│   ├── llm.py
│   ├── ollama_client.py
│   ├── pdf_processor.py
│   ├── rag_pipeline.py
│   ├── sparse_retriever.py
│   └── vector_store.py
│
└── test/
```

---

# 📌 Module Responsibilities

| Module                | Responsibility                                                               |
| --------------------- | ---------------------------------------------------------------------------- |
| `app.py`              | Streamlit UI, document upload, chat interface, statistics and source display |
| `pdf_processor.py`    | Extracts text from PDF pages                                                 |
| `chunker.py`          | Splits extracted document content into chunks                                |
| `embeddings.py`       | Handles embedding generation                                                 |
| `vector_store.py`     | Handles FAISS vector storage and similarity retrieval                        |
| `sparse_retriever.py` | Performs BM25 keyword retrieval                                              |
| `hybrid_retriever.py` | Combines dense and sparse retrieval                                          |
| `rag_pipeline.py`     | Coordinates retrieval, context construction and answer generation            |
| `llm.py`              | LLM-related functionality                                                    |
| `ollama_client.py`    | Handles communication with the local Ollama runtime                          |
| `evaluator.py`        | Retrieval/evaluation functionality                                           |

---

# ⚙️ Installation and Setup

## Prerequisites

Before running InsightForge AI, install:

* Python 3.10 or later
* Git
* Ollama
* Required Python packages
* Sufficient system resources to run the selected embedding model and LLM

---

# 1. Clone the Repository

Open PowerShell or a terminal:

```bash
git clone https://github.com/Lavanya5-12/InsightForge-AI.git
```

Move into the project directory:

```bash
cd InsightForge-AI
```

---

# 2. Create a Virtual Environment

On Windows:

```bash
python -m venv venv
```

Activate the virtual environment:

```bash
venv\Scripts\activate
```

After successful activation, you should see something similar to:

```text
(venv)
```

in the terminal.

---

# 3. Upgrade pip

```bash
python -m pip install --upgrade pip
```

---

# 4. Install Python Dependencies

Install the packages required by the project:

```bash
pip install -r requirements.txt
```

Wait for the installation to complete before starting the application.

---

# 5. Install and Configure Ollama

Make sure Ollama is installed and available on your system.

Verify the installation:

```bash
ollama --version
```

Pull the required Llama model:

```bash
ollama pull llama3.2:3b
```

Verify that the model is available:

```bash
ollama list
```

You should see the Llama model in the available models.

You can also test the model directly:

```bash
ollama run llama3.2:3b
```

If the model responds, the local LLM setup is working.

---

# ▶️ Running the Application

Make sure the virtual environment is activated.

From the project root directory, run:

```bash
streamlit run app.py
```

Streamlit will start the application and display a local URL.

Typically:

```text
http://localhost:8501
```

Open the URL in your browser.

---

# 🚀 Quick Start

For users who already have Python, Ollama and Git configured:

```bash
git clone https://github.com/Lavanya5-12/InsightForge-AI.git

cd InsightForge-AI

python -m venv venv

venv\Scripts\activate

pip install -r requirements.txt

ollama pull llama3.2:3b

streamlit run app.py
```

Then open:

```text
http://localhost:8501
```

---

# 🧪 How to Use the Application

## Step 1 — Upload PDF Documents

Open the Streamlit application.

Use the sidebar to upload one or more PDF documents.

The application supports multiple PDF files.

---

## Step 2 — Wait for Processing

After uploading documents, the application automatically processes them.

The pipeline is:

```text
PDF Upload
    ↓
Text Extraction
    ↓
Chunking
    ↓
Embedding Generation
    ↓
FAISS Index
    ↓
BM25 Index
    ↓
RAG Pipeline Ready
```

The sidebar displays document statistics including:

* Number of documents
* Number of pages
* Number of chunks

---

## Step 3 — Ask Questions

Once the documents are processed, enter a question related to their content.

Example:

```text
What are the main concepts discussed in this document?
```

Or:

```text
Explain the key points mentioned in Chapter 2.
```

---

## Step 4 — Hybrid Retrieval

The system performs both:

```text
Dense Retrieval
+
BM25 Retrieval
```

The configured retrieval weights are:

```text
Dense = 60%
BM25  = 40%
```

The most relevant chunks are selected.

---

## Step 5 — Answer Generation

The retrieved context is passed to:

```text
Llama 3.2:3b
```

through:

```text
Ollama
```

The model then generates the response.

---

## Step 6 — View Sources

Expand:

```text
📚 View Sources & Retrieval Scores
```

to inspect the supporting retrieval information.

This can include:

* Source document
* Page number
* Chunk number
* Dense score
* BM25 score
* Retrieval information

---

# 🔧 Troubleshooting

## Issue 1 — `streamlit` is not recognized

Make sure the virtual environment is activated:

```bash
venv\Scripts\activate
```

Then install the dependencies:

```bash
pip install -r requirements.txt
```

You can also run Streamlit through Python:

```bash
python -m streamlit run app.py
```

---

## Issue 2 — Ollama connection error

First check whether Ollama is available:

```bash
ollama --version
```

Then check the installed models:

```bash
ollama list
```

If Llama 3.2:3b is not present:

```bash
ollama pull llama3.2:3b
```

Test the model:

```bash
ollama run llama3.2:3b
```

Make sure Ollama is running before using the application.

---

## Issue 3 — Python import/package error

Make sure the virtual environment is activated:

```bash
venv\Scripts\activate
```

Then reinstall dependencies:

```bash
pip install -r requirements.txt
```

---

## Issue 4 — PDF text is missing

The current PDF extraction approach works best with **text-based PDFs**.

Scanned or image-only PDFs may not contain extractable text.

OCR support can be added as a future enhancement.

---

## Issue 5 — Poor retrieval results

If the retrieved chunks are not relevant, possible causes include:

* Poor PDF text extraction
* Chunking configuration
* Embedding quality
* BM25 matching
* Dense retrieval ranking
* Hybrid retrieval weighting
* Query formulation

A useful debugging process is:

```text
Check PDF Extraction
        ↓
Check Generated Chunks
        ↓
Check Dense Results
        ↓
Check BM25 Results
        ↓
Check Hybrid Results
        ↓
Check Final Context
        ↓
Check LLM Response
```

---

# 🔐 Privacy and Local Processing

InsightForge AI is designed around local document processing and local LLM inference.

The generation stage uses Llama 3.2:3b through Ollama instead of requiring an external LLM API.

This provides greater control over the processing environment and can be useful when working with documents that should remain on the local machine.

Users should still follow their organization's data-handling and security requirements when processing sensitive information.

---

# ⚠️ Current Limitations

The current implementation has several limitations:

1. It is primarily designed for text-based PDF documents.
2. Scanned/image-only PDFs may require OCR.
3. Llama 3.2:3b has lower capabilities than larger language models.
4. Retrieval quality depends on the quality of text extraction, chunking, embeddings, and ranking.
5. The 60/40 hybrid weighting may need tuning for different document types.
6. RAG reduces hallucination risk but cannot guarantee zero hallucinations.
7. FAISS is suitable for the current application scale but does not provide all the operational capabilities of a production-scale vector database.
8. Streamlit is primarily being used as the application's interactive interface and prototype layer.
9. Large-scale document collections may require additional storage, indexing, caching, and scalability improvements.

---

# 🚀 Future Enhancements

## Retrieval Improvements

* Cross-encoder reranking
* Query rewriting
* Query expansion
* Adaptive Top-K retrieval
* Improved hybrid score fusion
* Metadata-based filtering

## Document Processing

* OCR for scanned PDFs
* Table extraction
* Image-aware document processing
* Support for additional document formats

## RAG Improvements

* Contextual compression
* Multi-query retrieval
* Conversational memory improvements
* Citation verification
* Improved grounding checks

## Evaluation Improvements

* Larger evaluation datasets
* Automated benchmark testing
* Precision@K
* Recall@K
* MRR
* Faithfulness evaluation
* Answer relevance evaluation

## Production Improvements

* Persistent vector database
* Authentication
* Multi-user support
* Document management
* Monitoring and logging
* Caching
* Scalable model serving

## Advanced RAG

Future versions could explore:

* Multimodal RAG
* Hierarchical RAG
* Graph RAG
* Agentic RAG

---

# 🧪 Testing

The project contains a `test` directory for application testing.

Run the available tests using:

```bash
pytest
```

If pytest is not installed:

```bash
pip install pytest
```

Then run:

```bash
pytest
```

---

# 📊 Key Technical Concepts Demonstrated

This project demonstrates practical implementation of:

* Retrieval-Augmented Generation (RAG)
* Large Language Models
* Text embeddings
* Vector similarity search
* FAISS
* BM25
* Dense retrieval
* Sparse retrieval
* Hybrid retrieval
* Document chunking
* Context construction
* Local LLM inference
* Ollama
* Streamlit
* PDF processing
* Source attribution
* Retrieval scores
* RAG evaluation
* Python modular architecture
* Git/GitHub version control

---

# 💡 Why Hybrid Retrieval?

The main technical design decision in InsightForge AI is the combination of **dense semantic retrieval and sparse keyword retrieval**.

```text
                     User Question
                           │
             ┌─────────────┴─────────────┐
             ▼                           ▼
      Dense Retrieval              BM25 Retrieval
          FAISS                       BM25
             │                           │
             │        60%                │ 40%
             └─────────────┬─────────────┘
                           ▼
                    Hybrid Ranking
                           │
                           ▼
                   Relevant Chunks
                           │
                           ▼
                    Context Builder
                           │
                           ▼
                    Llama 3.2:3b
                           │
                           ▼
                       Answer
```

Dense retrieval helps answer:

> **"What information has a similar meaning?"**

BM25 helps answer:

> **"Where are the important/exact terms?"**

Using both provides a more balanced retrieval strategy.

---

# 🎓 Learning Outcomes

Through the development of InsightForge AI, the following practical concepts were implemented and explored:

1. Designing an end-to-end RAG pipeline.
2. Processing unstructured PDF documents.
3. Extracting and preserving document metadata.
4. Splitting documents into retrievable chunks.
5. Generating text embeddings.
6. Building a FAISS vector index.
7. Implementing semantic similarity retrieval.
8. Implementing BM25 keyword retrieval.
9. Combining dense and sparse retrieval.
10. Constructing context for an LLM.
11. Integrating a local Llama model using Ollama.
12. Building an interactive Streamlit application.
13. Displaying source and retrieval information.
14. Understanding retrieval evaluation.
15. Identifying limitations and failure points in RAG systems.
16. Structuring an AI application into modular Python components.

---

# 🔗 Repository

**GitHub Repository:**

https://github.com/Lavanya5-12/InsightForge-AI

---

# 👩‍💻 Author

**Sai Lavanya Neelagiri**

**Role:** AI Data Engineer Intern

**Project:** InsightForge AI

---

# ⭐ Conclusion

InsightForge AI demonstrates a practical implementation of a **local Hybrid RAG document intelligence system**.

The system combines:

```text
PDF Processing
      +
Text Chunking
      +
mxbai-embed-large
      +
FAISS
      +
BM25
      +
Hybrid Retrieval
      +
Llama 3.2:3b
      +
Ollama
      +
Streamlit
```

Instead of depending entirely on an LLM's pretrained knowledge, InsightForge AI retrieves relevant information from user-provided documents and uses that information as context for answer generation.

The project provides a foundation for extending document intelligence systems toward more advanced approaches such as **reranking, multimodal RAG, hierarchical retrieval, Graph RAG, and production-scale RAG architectures**.
