import os
from pathlib import Path
from pypdf import PdfWriter, PdfReader


def generate_fixture_pdf():
    fixture_dir = Path("data/fixtures")
    fixture_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = fixture_dir / "insightforge_test.pdf"

    pages_content = {
        1: (
            "InsightForge AI Test Document - Page 1\n"
            "Artificial Intelligence (AI) is a branch of computer science focused on building smart machines capable of "
            "performing tasks that typically require human intelligence, such as reasoning, learning, and problem-solving. "
            "The characteristics of Artificial Intelligence include adaptability, pattern recognition, decision-making capabilities, "
            "automated processing, and continuous learning from past data."
        ),
        2: (
            "InsightForge AI Test Document - Page 2\n"
            "What are the types of Artificial Intelligence?\n"
            "The main types of Artificial Intelligence include Narrow AI (Weak AI) which is designed for specific domain tasks, "
            "General AI (Strong AI) which possesses human-level cognitive capabilities across diverse tasks, "
            "and Superintelligent AI which theoretically surpasses human intelligence in all aspects."
        ),
        3: (
            "InsightForge AI Test Document - Page 3\n"
            "Machine Learning Methods\n"
            "Machine Learning includes supervised learning, unsupervised learning, and reinforcement learning. "
            "Supervised learning trains models on labeled datasets, while unsupervised learning discovers hidden patterns in unlabeled data."
        ),
        4: (
            "InsightForge AI Test Document - Page 4\n"
            "Vector Databases and Semantic Search\n"
            "Dense retrieval models convert text chunks into high-dimensional vector embeddings. "
            "FAISS (Facebook AI Similarity Search) enables high-performance vector search for semantic document retrieval."
        ),
        5: (
            "InsightForge AI Test Document - Page 5\n"
            "What is the history of Artificial Intelligence?\n"
            "The history of Artificial Intelligence began in the mid-20th century. Alan Turing proposed the Turing Test in 1950. "
            "In 1956, John McCarthy coined the term Artificial Intelligence at the Dartmouth Conference. "
            "The field evolved from early symbolic AI and expert systems to statistical machine learning and modern deep neural networks."
        ),
        6: (
            "InsightForge AI Test Document - Page 6\n"
            "What is an intelligent system?\n"
            "An intelligent system is a system that perceives its environment through sensors, processes information using reasoning rules "
            "or learning algorithms, takes rational actions to maximize success, and adapts its performance over time."
        ),
        7: (
            "InsightForge AI Test Document - Page 7\n"
            "Knowledge Representation\n"
            "Knowledge representation in AI involves structuring human knowledge using formal logics, ontologies, semantic networks, "
            "and knowledge graphs so intelligent agents can perform automated reasoning."
        ),
        8: (
            "InsightForge AI Test Document - Page 8\n"
            "What are the applications of Artificial Intelligence?\n"
            "The applications of Artificial Intelligence span healthcare medical diagnostics, autonomous transportation, "
            "financial fraud detection, natural language processing, robotics, automated document intelligence, and personalized recommendation systems."
        ),
        9: (
            "InsightForge AI Test Document - Page 9\n"
            "Computer Vision\n"
            "Computer vision algorithms process visual data from images and videos for object recognition, image segmentation, "
            "and automated spatial perception."
        ),
        10: (
            "InsightForge AI Test Document - Page 10\n"
            "Search Algorithms\n"
            "Artificial intelligence relies on heuristic search algorithms such as A* search, minimax algorithm, and Monte Carlo tree search."
        ),
        11: (
            "InsightForge AI Test Document - Page 11\n"
            "What are the current trends in Artificial Intelligence?\n"
            "The current trends in Artificial Intelligence include Large Language Models (LLMs), Retrieval-Augmented Generation (RAG), "
            "local edge AI inference, multimodal foundation models, autonomous agentic workflows, and AI alignment."
        ),
        12: (
            "InsightForge AI Test Document - Page 12\n"
            "AI Ethics and Governance\n"
            "Responsible AI requires fairness, accountability, transparency, data privacy protection, and robustness against algorithmic bias."
        ),
        13: (
            "InsightForge AI Test Document - Page 13\n"
            "Robotics and Physical Systems\n"
            "Robotics integrates physical actuators with AI perception and motor planning to operate in physical environments."
        ),
        14: (
            "InsightForge AI Test Document - Page 14\n"
            "What are the characteristics of problems in Artificial Intelligence?\n"
            "The characteristics of problems in Artificial Intelligence include high state-space complexity, incomplete or uncertain information, "
            "dynamic environments, non-deterministic state transitions, and multi-objective optimization trade-offs."
        ),
        15: (
            "InsightForge AI Test Document - Page 15\n"
            "Summary\n"
            "InsightForge AI provides a local hybrid RAG architecture combining FAISS vector search and BM25 sparse retrieval."
        )
    }

    # Construct clean PDF file manually
    body_parts = []
    
    # 1 0 obj: Catalog
    body_parts.append("1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n")
    
    num_pages = len(pages_content)
    page_ids = [3 + i * 2 for i in range(num_pages)]
    kids_str = " ".join(f"{pid} 0 R" for pid in page_ids)
    
    # 2 0 obj: Pages
    body_parts.append(f"2 0 obj\n<< /Type /Pages /Kids [{kids_str}] /Count {num_pages} >>\nendobj\n")
    
    font_id = 3 + num_pages * 2
    
    for i, page_num in enumerate(sorted(pages_content.keys())):
        text = pages_content[page_num]
        page_id = 3 + i * 2
        content_id = page_id + 1
        
        # Build text stream with Tj and T*
        lines = text.split("\n")
        stream_lines = ["BT /F1 12 Tf 50 750 Td 15 TL"]
        for line in lines:
            escaped = line.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
            stream_lines.append(f"({escaped}) Tj T*")
        stream_lines.append("ET")
        stream_str = "\n".join(stream_lines)
        
        page_obj = (
            f"{page_id} 0 obj\n"
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            f"/Resources << /Font << /F1 {font_id} 0 R >> >> /Contents {content_id} 0 R >>\n"
            f"endobj\n"
        )
        content_obj = (
            f"{content_id} 0 obj\n"
            f"<< /Length {len(stream_str)} >>\n"
            f"stream\n{stream_str}\nendstream\n"
            f"endobj\n"
        )
        
        body_parts.append(page_obj)
        body_parts.append(content_obj)
        
    font_obj = f"{font_id} 0 obj\n<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>\nendobj\n"
    body_parts.append(font_obj)
    
    header = "%PDF-1.4\n"
    offsets = []
    current_offset = len(header.encode("latin1"))
    
    full_body_bytes = bytearray()
    for part in body_parts:
        offsets.append(current_offset)
        part_bytes = part.encode("latin1")
        full_body_bytes.extend(part_bytes)
        current_offset += len(part_bytes)
        
    xref_offset = current_offset
    total_objects = len(offsets) + 1
    
    xref_lines = [f"xref\n0 {total_objects}\n0000000000 65535 f \n"]
    for off in offsets:
        xref_lines.append(f"{off:010d} 00000 n \n")
        
    xref_str = "".join(xref_lines)
    trailer_str = f"trailer\n<< /Size {total_objects} /Root 1 0 R >>\nstartxref\n{xref_offset}\n%%EOF\n"
    
    pdf_content = header.encode("latin1") + full_body_bytes + xref_str.encode("latin1") + trailer_str.encode("latin1")
    
    with open(pdf_path, "wb") as f:
        f.write(pdf_content)

    print(f"Generated clean test PDF: {pdf_path} ({len(pdf_content)} bytes, {num_pages} pages)")


if __name__ == "__main__":
    generate_fixture_pdf()
