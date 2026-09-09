# 🧠 InsightForge AI

## Hybrid RAG Document Intelligence System

InsightForge AI is a **local-first Hybrid Retrieval-Augmented Generation (RAG) document question-answering application**. It allows users to upload PDF documents and ask natural-language questions grounded strictly in document content.

The system combines **Dense Vector Retrieval (FAISS)** with **Sparse Keyword Retrieval (BM25)** using a weighted hybrid score (**60% Dense / 40% BM25**), backed by a local **Ollama** runtime running **Llama 3.2:3b** for answer generation and **mxbai-embed-large** for vector embeddings.

---

## ⚡ Quick Start (Windows / Linux / macOS)

### 1. Clone the Repository
```bash
git clone https://github.com/Lavanya5-12/InsightForge-AI.git
cd InsightForge-AI
```

### 2. Set Up Virtual Environment
```bash
python -m venv venv
# On Windows PowerShell:
.\venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Set Up Local Ollama Models
Ensure [Ollama](https://ollama.com) is installed and running, then pull the required models:
```bash
ollama pull mxbai-embed-large:latest
ollama pull llama3.2:3b
```
Verify installed models:
```bash
ollama list
```

### 5. Launch the Application
```bash
streamlit run app.py
```
Open browser at: `http://localhost:8501`

---

## 📌 Project Overview & Objectives

Traditional keyword search fails when user queries use different terminology than the target document. Conversely, pure vector search can miss exact technical terms, product codes, or named entities.

InsightForge AI resolves these limitations through:
- 📄 **Metadata-Preserving Extraction & Chunking**: Preserves document name, page number, and chunk index across every stage.
- 📐 **Full Ingestion Pipeline**: Connects PDF extraction $\rightarrow$ chunking $\rightarrow$ embedding generation $\rightarrow$ FAISS vector indexing + BM25 indexing.
- 🔀 **Hybrid Retrieval Fusion**: Combines 60% dense vector similarity (FAISS) and 40% sparse keyword matching (BM25).
- 🛡️ **Retrieval Quality Threshold**: Evaluates retrieval relevance (`RETRIEVAL_THRESHOLD = 0.35`). Rejects out-of-scope or unsupported queries before calling the LLM.
- 📚 **Dynamic Citations**: Automatically generates verifiable document and page citations (e.g., `document.pdf — Page 3`).
- 🔬 **Reproducible Test & Evaluation Suite**: Includes deterministic test fixture `data/fixtures/insightforge_test.pdf`, automated pytest suite, and evaluation script.

---

## 🏗️ System Architecture

```text
               📄 UPLOADED PDF DOCUMENTS
                         │
                         ▼
                ┌─────────────────┐
                │ PDF Text        │
                │ Extraction      │
                └────────┬────────┘
                         │
                         ▼
                ┌─────────────────┐
                │ Metadata        │
                │ Chunking        │
                └────────┬────────┘
                         │
                         ▼
           ┌─────────────┴─────────────┐
           ▼                           ▼
 ┌───────────────────┐       ┌───────────────────┐
 │ Embedding Gen     │       │ Tokenization      │
 │ mxbai-embed-large │       │ BM25 Tokenizer    │
 └─────────┬─────────┘       └─────────┬─────────┘
           │                           │
           ▼                           ▼
 ┌───────────────────┐       ┌───────────────────┐
 │ FAISS Vector      │       │ BM25 Sparse       │
 │ Store (Dense)     │       │ Store             │
 └───────────────────┘       └───────────────────┘


                   🔍 USER QUESTION
                         │
             ┌───────────┴───────────┐
             ▼                       ▼
      FAISS Dense Search       BM25 Keyword Search
      Vector Similarity        Term Frequency
             │                       │
             └───────────┬───────────┘
                         ▼
               Normalized Hybrid Score
                (60% Dense + 40% BM25)
                         │
                         ▼
             🛡️ Relevance Threshold Check
               (Score >= 0.35 Threshold)
             ┌───────────┴───────────┐
      Below Threshold          Above Threshold
             │                       │
             ▼                       ▼
     Grounded Fallback       Context Construction
     "Info Not Found"                │
                                     ▼
                            Ollama Llama 3.2:3b
                                     │
                                     ▼
                              Final Response +
                              Dynamic Citations
```

---

## 🔄 Ingestion & Retrieval Pipeline

### 1. Document Ingestion
1. PDF uploaded via Streamlit interface or script.
2. `src/pdf_processor.py` extracts page text and attaches `page_number` and `document_name`.
3. `src/chunker.py` splits text into overlapping word chunks (default: 500 words, 100 overlap).
4. `src/embeddings.py` calls Ollama (`mxbai-embed-large:latest`) to generate vector embeddings for all chunks.
5. `src/vector_store.py` builds the FAISS L2 vector index (`index.ntotal == len(chunks)`).
6. `src/sparse_retriever.py` initializes BM25 token index.

### 2. Hybrid Retrieval & Score Fusion
- **Dense Score**: FAISS similarity metric: $S_{\text{dense}} = \frac{1}{1 + D_{\text{L2}}}$.
- **Sparse Score**: BM25 keyword score normalized by top BM25 match score $S_{\text{sparse}} = \frac{\text{BM25}}{\text{max(BM25)}}$.
- **Hybrid Score**:
  $$S_{\text{hybrid}} = 0.6 \times S_{\text{dense}} + 0.4 \times S_{\text{sparse}}$$

### 3. Safety Threshold & Out-of-Scope Protection
- Configured in `src/config.py` as `RETRIEVAL_THRESHOLD = 0.35`.
- If $S_{\text{hybrid}} < 0.35$ for all chunks:
  - System returns: *"I couldn't find sufficient information about this question in the uploaded documents."*
  - Prevents hallucinated LLM responses.

---

## 🛠️ Technology Stack

| Layer | Component / Tool | Reason Selected |
|---|---|---|
| **Language** | Python 3.10+ | Standard ecosystem for AI/RAG development |
| **Frontend** | Streamlit | Lightweight interactive web UI |
| **PDF Extraction** | `pypdf` | Fast, local PDF parser with page metadata |
| **Dense Vector Store**| FAISS (`faiss-cpu`) | High-performance vector similarity search |
| **Sparse Retrieval** | `rank-bm25` | Industry-standard BM25 retrieval implementation |
| **Embedding Model** | `mxbai-embed-large:latest` | High-quality 1024-dim open local embedding model |
| **LLM Engine** | `llama3.2:3b` via Ollama | Local-first open-source 3B instruction model |
| **Test Suite** | `pytest` | Standard Python testing framework |

---

## ⚙️ Configuration (`src/config.py`)

Centralized configuration values:
```python
OLLAMA_BASE_URL = "http://localhost:11434"
EMBEDDING_MODEL = "mxbai-embed-large:latest"
LLM_MODEL = "llama3.2:3b"
DEFAULT_CHUNK_SIZE = 500
DEFAULT_CHUNK_OVERLAP = 100
DEFAULT_TOP_K = 5
DENSE_WEIGHT = 0.6
SPARSE_WEIGHT = 0.4
RETRIEVAL_THRESHOLD = 0.35
```

---

## 🧪 Testing & Evaluation

### Run Automated Pytest Suite
```bash
python -m pytest
```
Runs tests covering PDF extraction, chunking, FAISS index management, BM25 retrieval, hybrid score fusion, thresholding, out-of-scope rejection, and evaluator precision.

### Run Reproducible Evaluation Benchmark
```bash
python test/evaluate_rag.py
```
This script automatically generates the test fixture `data/fixtures/insightforge_test.pdf`, runs evaluation queries, and saves results to `evaluation/results.json` and `evaluation/results.md`.

---

## 📊 Evaluation Results

Benchmark results on `data/fixtures/insightforge_test.pdf` (15 pages, 15 chunks):

| Metric | Score |
|---|---|
| **Total Questions Evaluated** | 8 |
| **Hit Rate (Recall@5 >= 1)** | **100.0%** (8/8) |
| **Overall Precision@5** | **0.2000** |
| **Average Top Hybrid Score** | **0.9354** |

Saved in [`evaluation/results.md`](file:///c:/Users/NeelagiriSaiLavanya/Desktop/ps/evaluation/results.md) and [`evaluation/results.json`](file:///c:/Users/NeelagiriSaiLavanya/Desktop/ps/evaluation/results.json).

---

## ❓ Troubleshooting

1. **Ollama Connection Error**:
   - Ensure Ollama is running: `ollama serve`
   - Test health endpoint: `curl http://localhost:11434/api/tags`
2. **Model Missing Error**:
   - Run `ollama pull mxbai-embed-large:latest` and `ollama pull llama3.2:3b`.
3. **ModuleNotFoundError**:
   - Always run commands with `python -m ...` or ensure virtual environment is active.
