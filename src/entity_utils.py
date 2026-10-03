"""
Entity-aware query expansion, normalization, and verification helper functions for RAG retrieval.
Supports factual/entity queries (telephone, email, CAS number, supplier, address, product, molecular weight, formula, emergency).
"""

import re
from typing import List, Dict, Set, Optional

# Pre-compiled regex pattern for whitespace cleaning
WHITESPACE_RE = re.compile(r"\s+")

# Common entity categories and definitions
ENTITY_DEFINITIONS: Dict[str, Dict] = {
    "telephone": {
        "keywords": [
            "telephone", "telephone number", "tel", "tele", "phone",
            "phone number", "contact number", "contact phone", "fax",
            "tele number", "tele phone", "phone no", "tel no"
        ],
        "expansion_terms": ["telephone", "tel", "phone", "contact", "fax"],
        "labels": [
            "tel:", "tel", "telephone:", "telephone", "phone:", "phone",
            "phone no:", "telephone no:", "telephone number:", "contact:",
            "contact number:", "tel.:", "tel.", "fax:", "fax"
        ],
        "patterns": [
            r"\b(tel|telephone|phone|fax|contact)\b\s*[:\.]?\s*\+?[\d\s\-\(\)\.]{7,}",
            r"\+?\d{1,3}[\s\-\.]?\(?\d{2,4}\)?[\s\-\.]?\d{3,4}[\s\-\.]?\d{3,4}"
        ]
    },
    "email": {
        "keywords": [
            "email", "e-mail", "email address", "mail address"
        ],
        "expansion_terms": ["email", "e-mail", "mail"],
        "labels": ["email:", "e-mail:", "email", "e-mail", "mail:"],
        "patterns": [
            r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"
        ]
    },
    "cas": {
        "keywords": [
            "cas", "cas number", "cas no", "cas rn", "chemical abstracts service"
        ],
        "expansion_terms": ["cas", "cas no", "cas number", "cas rn"],
        "labels": ["cas:", "cas no:", "cas number:", "cas no", "cas rn:", "cas#"],
        "patterns": [
            r"\bcas\b\s*[:\.\-]?\s*(?:no|number|rn)?\s*[:\.]?\s*\d{2,7}-\d{2}-\d\b",
            r"\b\d{2,7}-\d{2}-\d\b"
        ]
    },
    "supplier": {
        "keywords": [
            "supplier", "manufacturer", "company", "vendor"
        ],
        "expansion_terms": ["supplier", "manufacturer", "company", "vendor"],
        "labels": ["supplier:", "manufacturer:", "company:", "vendor:", "manufactured by:", "distributed by:"],
        "patterns": [
            r"\b(supplier|manufacturer|company|vendor)\b\s*[:\.]?"
        ]
    },
    "address": {
        "keywords": [
            "address", "location", "company address", "supplier address"
        ],
        "expansion_terms": ["address", "location", "street", "road", "zip", "city", "pa", "usa"],
        "labels": ["address:", "location:", "company address:", "supplier address:"],
        "patterns": [
            r"\baddress\b\s*[:\.]?",
            r"\b\d+\s+[A-Za-z0-9\s,\.]+(?:road|rd|street|st|avenue|ave|drive|drv|boulevard|blvd|way|lane|ln)\b",
            r"\b[A-Z][a-zA-Z\s]+,\s*[A-Z]{2}\s+\d{5}(?:-\d{4})?\b"
        ]
    },
    "product": {
        "keywords": [
            "product", "product name", "substance name", "chemical name"
        ],
        "expansion_terms": ["product", "product name", "substance", "chemical name"],
        "labels": ["product name:", "product:", "substance name:", "trade name:", "chemical name:"],
        "patterns": [
            r"\b(product name|product|substance name|trade name|chemical name)\b\s*[:\.]?"
        ]
    },
    "molecular_weight": {
        "keywords": [
            "molecular weight", "molecular mass", "mol wt", "mw"
        ],
        "expansion_terms": ["molecular weight", "molecular mass", "mol wt", "mw"],
        "labels": ["molecular weight:", "mol wt:", "mw:", "g/mol"],
        "patterns": [
            r"\b(molecular weight|mol wt|mw)\b\s*[:\.]?"
        ]
    },
    "formula": {
        "keywords": [
            "formula", "molecular formula", "chemical formula"
        ],
        "expansion_terms": ["formula", "molecular formula", "chemical formula"],
        "labels": ["formula:", "molecular formula:", "chemical formula:"],
        "patterns": [
            r"\b(formula|molecular formula|chemical formula)\b\s*[:\.]?"
        ]
    },
    "emergency": {
        "keywords": [
            "emergency number", "emergency telephone", "emergency phone", "emergency contact", "chemtrec"
        ],
        "expansion_terms": ["emergency", "emergency number", "emergency phone", "emergency contact"],
        "labels": ["emergency phone:", "emergency tel:", "emergency contact:", "emergency number:"],
        "patterns": [
            r"\b(emergency)\b\s*(phone|telephone|contact|number|tel)?\s*[:\.]?"
        ]
    }
}


def normalize_query(query: str) -> str:
    """Normalize query text for entity matching."""
    if not query:
        return ""
    clean = query.lower().strip()
    return WHITESPACE_RE.sub(" ", clean)


def detect_entity_types(query: str) -> List[str]:
    """
    Recognize factual/entity intent in the user's query.
    Returns list of matched entity categories (e.g. ['telephone']).
    """
    normalized = normalize_query(query)
    if not normalized:
        return []

    words = set(re.findall(r"\b\w+\b", normalized))
    matched_entities: List[str] = []

    for entity_type, defs in ENTITY_DEFINITIONS.items():
        # Check keyword phrases first
        phrase_match = any(kw in normalized for kw in defs["keywords"])
        if phrase_match:
            matched_entities.append(entity_type)
            continue

        # Check single-word keyword matches
        for kw in defs["keywords"]:
            kw_words = kw.split()
            if len(kw_words) == 1 and kw in words:
                matched_entities.append(entity_type)
                break

    return matched_entities


def expand_query_for_bm25(query: str, entity_types: Optional[List[str]] = None) -> str:
    """
    Expand query with relevant terms for BM25 sparse retrieval.
    Original query is preserved, and expansion terms are appended.
    """
    if entity_types is None:
        entity_types = detect_entity_types(query)

    if not entity_types:
        return query

    expansion_terms: Set[str] = set()
    for et in entity_types:
        if et in ENTITY_DEFINITIONS:
            expansion_terms.update(ENTITY_DEFINITIONS[et]["expansion_terms"])

    if not expansion_terms:
        return query

    extra_str = " ".join(sorted(expansion_terms))
    return f"{query} {extra_str}"


def has_entity_evidence(chunk_text: str, entity_types: List[str]) -> bool:
    """
    Check if chunk text contains actual lexical or pattern evidence for the requested entity types.
    Prevents false fallback promotion when no evidence exists in the chunk.
    """
    if not chunk_text or not entity_types:
        return False

    text_lower = chunk_text.lower()

    for et in entity_types:
        if et not in ENTITY_DEFINITIONS:
            continue

        defs = ENTITY_DEFINITIONS[et]

        # 1. Check labels
        label_match = any(label in text_lower for label in defs["labels"])

        # 2. Check regex patterns
        pattern_match = False
        for pat in defs["patterns"]:
            if re.search(pat, chunk_text, re.IGNORECASE):
                pattern_match = True
                break

        if label_match or pattern_match:
            return True

    return False
