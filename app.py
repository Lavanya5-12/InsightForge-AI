import os
import streamlit as st

from src.pdf_processor import extract_pdf_text, PDFProcessingError
from src.chunker import chunk_text
from src.rag_pipeline import RAGPipeline
from src.ollama_client import default_ollama_client, OllamaUnavailableError, OllamaModelNotFoundError
from src.config import EMBEDDING_MODEL, LLM_MODEL, RETRIEVAL_THRESHOLD, DENSE_WEIGHT, SPARSE_WEIGHT


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="InsightForge AI",
    page_icon="🧠",
    layout="wide"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>
    .main-title {
        font-size: 38px;
        font-weight: 700;
        margin-bottom: 0px;
    }
    .subtitle {
        font-size: 16px;
        color: #666666;
        margin-bottom: 20px;
    }
    .citation-box {
        background-color: #f8f9fa;
        padding: 12px;
        border-radius: 6px;
        border-left: 4px solid #4CAF50;
        margin-top: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# HEADER
# ============================================================

st.markdown('<div class="main-title">🧠 InsightForge AI</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Local Hybrid RAG Document Intelligence (FAISS + BM25 + Llama 3.2)</div>', unsafe_allow_html=True)


# ============================================================
# OLLAMA HEALTH CHECK AT STARTUP
# ============================================================

ollama_online, ollama_msg = default_ollama_client.check_health()
if not ollama_online:
    st.error(f"⚠️ **Ollama Service Offline**: {ollama_msg}")
    st.info("💡 **Fix**: Please start Ollama on your system (`ollama serve`) and refresh this page.")
else:
    embed_ok, embed_msg = default_ollama_client.check_model_available(EMBEDDING_MODEL)
    llm_ok, llm_msg = default_ollama_client.check_model_available(LLM_MODEL)
    if not embed_ok:
        st.warning(f"⚠️ Embedding model missing: {embed_msg}")
    if not llm_ok:
        st.warning(f"⚠️ LLM model missing: {llm_msg}")


# ============================================================
# SESSION STATE
# ============================================================

if "rag" not in st.session_state:
    st.session_state.rag = None

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "document_names" not in st.session_state:
    st.session_state.document_names = []

if "total_pages" not in st.session_state:
    st.session_state.total_pages = 0

if "total_chunks" not in st.session_state:
    st.session_state.total_chunks = 0

if "total_vectors" not in st.session_state:
    st.session_state.total_vectors = 0


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.header("📄 Documents")

    uploaded_files = st.file_uploader(
        "Upload PDF documents",
        type=["pdf"],
        accept_multiple_files=True
    )

    # ========================================================
    # PROCESS DOCUMENTS
    # ========================================================
    if uploaded_files:
        current_names = sorted([file.name for file in uploaded_files])
        previous_names = sorted(st.session_state.document_names)

        if current_names != previous_names:
            upload_dir = os.path.join("data", "uploads")
            os.makedirs(upload_dir, exist_ok=True)

            all_pages = []
            document_names = []
            total_pages = 0

            with st.spinner("Processing PDF text, chunking, and generating FAISS embeddings..."):
                try:
                    for uploaded_file in uploaded_files:
                        file_path = os.path.join(upload_dir, uploaded_file.name)

                        with open(file_path, "wb") as f:
                            f.write(uploaded_file.getbuffer())

                        pages = extract_pdf_text(file_path)
                        if not pages:
                            st.warning(f"No extractable text found in '{uploaded_file.name}'.")
                            continue

                        total_pages += len(pages)
                        all_pages.extend(pages)
                        document_names.append(uploaded_file.name)

                    if not all_pages:
                        st.error("No valid text could be extracted from the uploaded PDF documents.")
                    else:
                        # 1. Create chunks
                        chunks = chunk_text(all_pages)

                        # 2. Build RAG pipeline (connects chunks -> generate_embeddings -> add_documents -> FAISS)
                        st.session_state.rag = RAGPipeline(chunks, reset_vector_store=True)

                        # 3. Save statistics
                        st.session_state.document_names = document_names
                        st.session_state.total_pages = total_pages
                        st.session_state.total_chunks = len(chunks)
                        st.session_state.total_vectors = st.session_state.rag.vector_store.count()

                        # Clear previous chat on new corpus upload
                        st.session_state.chat_history = []
                        st.success(f"Successfully processed {len(document_names)} PDF(s) into {len(chunks)} chunks and {st.session_state.total_vectors} FAISS vectors!")

                except Exception as e:
                    st.error(f"Error during document processing: {str(e)}")

    # ========================================================
    # DOCUMENT STATISTICS & ANALYTICS
    # ========================================================
    if st.session_state.document_names:
        st.divider()
        st.subheader("📊 Document Statistics")

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Documents", len(st.session_state.document_names))
        with col2:
            st.metric("Pages", st.session_state.total_pages)

        col3, col4 = st.columns(2)
        with col3:
            st.metric("Chunks", st.session_state.total_chunks)
        with col4:
            st.metric("FAISS Vectors", st.session_state.total_vectors)

        st.divider()
        st.subheader("🔬 System Analytics")

        st.write("🔀 **Retrieval:** Hybrid (FAISS Dense + BM25 Sparse)")
        st.write(f"⚖️ **Weights:** {int(DENSE_WEIGHT*100)}% Dense / {int(SPARSE_WEIGHT*100)}% BM25")
        st.write(f"🛡️ **Threshold:** {RETRIEVAL_THRESHOLD}")
        st.write(f"📐 **Embeddings:** {EMBEDDING_MODEL}")
        st.write(f"🧠 **LLM:** {LLM_MODEL}")

        st.divider()
        st.subheader("📚 Loaded Documents")
        for name in st.session_state.document_names:
            st.write(f"📄 {name}")

        st.divider()
        if st.button("🗑️ Clear Chat History", use_container_width=True):
            st.session_state.chat_history = []
            st.rerun()


# ============================================================
# MAIN APPLICATION AREA
# ============================================================

if st.session_state.rag is None:
    st.info("👈 Upload one or more PDF documents from the sidebar to get started.")

else:
    st.success(f"📚 {len(st.session_state.document_names)} document(s) indexed and ready for questions.")

    # ========================================================
    # CHAT HISTORY DISPLAY
    # ========================================================
    if st.session_state.chat_history:
        st.subheader("💬 Conversation")

        for message in st.session_state.chat_history:
            with st.chat_message("user"):
                st.write(message["question"])

            with st.chat_message("assistant"):
                st.write(message["answer"])

                if message.get("sources"):
                    with st.expander("📚 View Retrieved Sources & Hybrid Scores"):
                        for i, source in enumerate(message["sources"], start=1):
                            metadata = source.get("metadata", {})
                            doc_name = metadata.get("document_name", "Unknown document")
                            page = metadata.get("page_number", "Unknown")
                            chunk_num = metadata.get("chunk_number", "Unknown")

                            dense_score = source.get("dense_score", 0.0)
                            sparse_score = source.get("sparse_score", 0.0)
                            hybrid_score = source.get("hybrid_score", 0.0)
                            reason = source.get("retrieval_reason", "hybrid")

                            st.markdown(f"#### 📚 Source {i} — {doc_name} (Page {page}, Chunk {chunk_num})")

                            sc1, sc2, sc3 = st.columns(3)
                            with sc1:
                                st.metric("Dense Score", f"{dense_score:.4f}")
                            with sc2:
                                st.metric("BM25 Score", f"{sparse_score:.4f}")
                            with sc3:
                                st.metric("Hybrid Score", f"{hybrid_score:.4f}")

                            st.progress(min(max(hybrid_score, 0.0), 1.0))
                            if reason != "hybrid":
                                st.caption(f"Mode: **{reason}** | Hybrid Score = {int(DENSE_WEIGHT*100)}% Dense + {int(SPARSE_WEIGHT*100)}% BM25")
                            else:
                                st.caption(f"Hybrid Score = {int(DENSE_WEIGHT*100)}% Dense + {int(SPARSE_WEIGHT*100)}% BM25")
                            st.markdown("**Content:**")
                            st.write(source.get("text", ""))

                            if i < len(message["sources"]):
                                st.divider()

    # ========================================================
    # QUESTION INPUT
    # ========================================================
    st.divider()
    st.subheader("🔍 Ask your documents")

    with st.form(key="question_form", clear_on_submit=False):
        question = st.text_input(
            "Enter your question",
            placeholder="Example: What are the main concepts discussed in this document?"
        )
        submit_button = st.form_submit_button("🔍 Ask Question", type="primary")

    if submit_button:
        if not question.strip():
            st.warning("Please enter a question.")
        else:
            with st.spinner("Searching FAISS + BM25, applying relevance threshold, and generating response..."):
                try:
                    result = st.session_state.rag.ask(question)

                    st.session_state.chat_history.append({
                        "question": question,
                        "answer": result["answer"],
                        "sources": result["sources"],
                        "evaluation": result.get("evaluation", {})
                    })
                    st.rerun()

                except (OllamaUnavailableError, OllamaModelNotFoundError) as oerr:
                    st.error(f"Ollama Error: {str(oerr)}")
                except Exception as err:
                    st.error(f"An error occurred while answering your question: {str(err)}")