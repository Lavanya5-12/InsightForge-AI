import os
import streamlit as st

from src.pdf_processor import extract_pdf_text
from src.chunker import chunk_text
from src.rag_pipeline import RAGPipeline


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
        font-size: 42px;
        font-weight: 700;
        margin-bottom: 0px;
    }

    .subtitle {
        font-size: 18px;
        color: #777777;
        margin-bottom: 25px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="main-title">🧠 InsightForge AI</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Hybrid RAG Document Intelligence'
    '</div>',
    unsafe_allow_html=True
)


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

        current_names = sorted(
            [file.name for file in uploaded_files]
        )

        previous_names = sorted(
            st.session_state.document_names
        )

        if current_names != previous_names:

            os.makedirs(
                "data/uploads",
                exist_ok=True
            )

            all_pages = []
            document_names = []
            total_pages = 0

            with st.spinner(
                "Processing documents..."
            ):

                for uploaded_file in uploaded_files:

                    file_path = os.path.join(
                        "data/uploads",
                        uploaded_file.name
                    )

                    with open(
                        file_path,
                        "wb"
                    ) as file:

                        file.write(
                            uploaded_file.getbuffer()
                        )


                    # ----------------------------------------
                    # Extract PDF
                    # ----------------------------------------

                    pages = extract_pdf_text(
                        file_path
                    )

                    total_pages += len(pages)


                    # ----------------------------------------
                    # Attach document name
                    # ----------------------------------------

                    for page in pages:

                        page["document_name"] = (
                            uploaded_file.name
                        )


                    all_pages.extend(
                        pages
                    )

                    document_names.append(
                        uploaded_file.name
                    )


                # --------------------------------------------
                # Create chunks
                # --------------------------------------------

                chunks = chunk_text(
                    all_pages
                )


                # --------------------------------------------
                # Create RAG pipeline
                # --------------------------------------------

                st.session_state.rag = (
                    RAGPipeline(chunks)
                )


                # --------------------------------------------
                # Save statistics
                # --------------------------------------------

                st.session_state.document_names = (
                    document_names
                )

                st.session_state.total_pages = (
                    total_pages
                )

                st.session_state.total_chunks = (
                    len(chunks)
                )


                # --------------------------------------------
                # Clear previous chat
                # --------------------------------------------

                st.session_state.chat_history = []


            st.success(
                "Documents processed successfully!"
            )


    # ========================================================
    # DOCUMENT STATISTICS
    # ========================================================

    if st.session_state.document_names:

        st.divider()

        st.subheader(
            "📊 Document Statistics"
        )


        col1, col2 = st.columns(2)


        with col1:

            st.metric(
                "Documents",
                len(
                    st.session_state.document_names
                )
            )


        with col2:

            st.metric(
                "Pages",
                st.session_state.total_pages
            )


        st.metric(
            "Chunks",
            st.session_state.total_chunks
        )


        # ====================================================
        # SYSTEM ANALYTICS
        # ====================================================

        st.divider()

        st.subheader(
            "🔬 System Analytics"
        )


        analytics_col1, analytics_col2 = st.columns(2)


        with analytics_col1:

            st.write(
                "🔀 **Retrieval**"
            )

            st.caption(
                "FAISS + BM25 Hybrid Retrieval"
            )


        with analytics_col2:

            st.write(
                "🧠 **Generation**"
            )

            st.caption(
                "Ollama — Llama 3.2:3b"
            )


        st.write(
            "📐 **Embedding Model**"
        )

        st.caption(
            "mxbai-embed-large"
        )


        # ====================================================
        # LOADED DOCUMENTS
        # ====================================================

        st.divider()

        st.subheader(
            "📚 Loaded Documents"
        )


        for name in st.session_state.document_names:

            st.write(
                f"📄 {name}"
            )


        # ====================================================
        # RAG CONFIGURATION
        # ====================================================

        st.divider()

        st.subheader(
            "⚙️ RAG Configuration"
        )


        st.write(
            "🔀 **Retrieval:** FAISS + BM25"
        )

        st.write(
            "⚖️ **Weights:** Dense 60% + BM25 40%"
        )

        st.write(
            "🧠 **LLM:** Llama 3.2:3b"
        )


        # ====================================================
        # CLEAR CHAT
        # ====================================================

        st.divider()


        if st.button(
            "🗑️ Clear Chat",
            use_container_width=True
        ):

            st.session_state.chat_history = []

            st.rerun()


# ============================================================
# MAIN APPLICATION
# ============================================================

if st.session_state.rag is None:

    st.info(
        "👈 Upload one or more PDF documents "
        "from the sidebar to get started."
    )


else:

    st.success(
        f"📚 "
        f"{len(st.session_state.document_names)} "
        f"document(s) ready for questions."
    )


    # ========================================================
    # CHAT HISTORY
    # ========================================================

    if st.session_state.chat_history:

        st.subheader(
            "💬 Conversation"
        )


        for message in st.session_state.chat_history:

            # ------------------------------------------------
            # USER MESSAGE
            # ------------------------------------------------

            with st.chat_message("user"):

                st.write(
                    message["question"]
                )


            # ------------------------------------------------
            # ASSISTANT MESSAGE
            # ------------------------------------------------

            with st.chat_message("assistant"):

                st.write(
                    message["answer"]
                )


                # --------------------------------------------
                # SOURCE INFORMATION
                # --------------------------------------------

                with st.expander(
                    "📚 View Sources & Retrieval Scores"
                ):


                    for i, source in enumerate(
                        message["sources"],
                        start=1
                    ):

                        metadata = source.get(
                            "metadata",
                            {}
                        )


                        document_name = metadata.get(
                            "document_name",
                            "Unknown document"
                        )


                        page = metadata.get(
                            "page_number",
                            "Unknown"
                        )


                        chunk_number = metadata.get(
                            "chunk_number",
                            "Unknown"
                        )


                        dense_score = source.get(
                            "dense_score",
                            0.0
                        )


                        sparse_score = source.get(
                            "sparse_score",
                            0.0
                        )


                        hybrid_score = source.get(
                            "hybrid_score",
                            0.0
                        )


                        # ------------------------------------
                        # Source heading
                        # ------------------------------------

                        st.markdown(
                            f"### 📚 Source {i}"
                        )


                        st.markdown(
                            f"📄 **Document:** "
                            f"{document_name}"
                        )

                        st.markdown(
                            f"📖 **Page:** {page}"
                        )

                        st.markdown(
                            f"🧩 **Chunk:** "
                            f"{chunk_number}"
                        )


                        # ------------------------------------
                        # Retrieval scores
                        # ------------------------------------

                        st.markdown(
                            "#### 📊 Retrieval Scores"
                        )


                        score_col1, score_col2, score_col3 = (
                            st.columns(3)
                        )


                        with score_col1:

                            st.metric(
                                "Dense Score",
                                f"{dense_score:.4f}"
                            )


                        with score_col2:

                            st.metric(
                                "BM25 Score",
                                f"{sparse_score:.4f}"
                            )


                        with score_col3:

                            st.metric(
                                "Hybrid Score",
                                f"{hybrid_score:.4f}"
                            )


                        # ------------------------------------
                        # Score explanation
                        # ------------------------------------

                        st.progress(
                            min(
                                max(
                                    hybrid_score,
                                    0.0
                                ),
                                1.0
                            )
                        )


                        st.caption(
                            "Hybrid score = "
                            "60% Dense + 40% BM25"
                        )


                        # ------------------------------------
                        # Retrieved text
                        # ------------------------------------

                        st.markdown(
                            "#### 📄 Retrieved Content"
                        )


                        st.write(
                            source["text"]
                        )


                        if i < len(
                            message["sources"]
                        ):

                            st.divider()


    else:

        st.info(
            "Ask your first question about "
            "the uploaded documents."
        )


    # ========================================================
    # QUESTION SECTION
    # ========================================================

    st.divider()

    st.subheader(
        "🔍 Ask your documents"
    )


    question = st.text_input(
        "Enter your question",
        placeholder=(
            "Example: What is Artificial Intelligence?"
        )
    )


    ask_button = st.button(
        "🔍 Ask",
        type="primary"
    )


    # ========================================================
    # QUESTION PROCESSING
    # ========================================================

    if ask_button:

        if not question.strip():

            st.warning(
                "Please enter a question."
            )


        else:

            with st.spinner(
                "Searching documents and generating answer..."
            ):

                result = st.session_state.rag.ask(
                    question,
                    n_results=5
                )


            # --------------------------------------------
            # Save conversation
            # --------------------------------------------

            st.session_state.chat_history.append(
                {
                    "question": question,
                    "answer": result["answer"],
                    "sources": result["sources"]
                }
            )


            # --------------------------------------------
            # Refresh
            # --------------------------------------------

            st.rerun()