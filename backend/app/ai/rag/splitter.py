"""
RAG Text Chunking & Splitting (backend/app/ai/rag/splitter.py)
-------------------------------------------------------------
PYTHON & NLP CONCEPTS DEMONSTRATED:
- Recursive Character Splitting: Progressively tries natural boundary separators
  (paragraphs -> sentences -> clauses -> words) to preserve coherent semantic units.
- Multilingual Punctuation Support: Respects Devanagari / Indian punctuation ('।', '॥')
  alongside standard Western delimiters ('.', '!', '?').
- Metadata Preservation: Embeds origin document ID, chunk index, page numbers,
  and character spans into LangChain Document records.
"""

from typing import Any, Dict, List, Optional
from langchain_core.documents import Document as LCDocument
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.core.config import settings
from app.ai.rag.parser import ParsedSection


class RAGTextSplitter:
    """
    Splits parsed document sections into overlapping semantic chunks optimized
    for dense vector embedding and LLM prompt context injection.
    """

    def __init__(
        self,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None
    ):
        self.chunk_size = chunk_size or settings.RAG_CHUNK_SIZE
        self.chunk_overlap = chunk_overlap or settings.RAG_CHUNK_OVERLAP

        # Hierarchical separators preserving semantic boundaries
        self.separators = [
            "\n\n",
            "\n",
            "।",   # Devanagari full stop
            "॥",   # Devanagari section terminator
            ". ",
            "? ",
            "! ",
            "; ",
            ", ",
            " ",
            ""
        ]

        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=self.separators,
            length_function=len,
            is_separator_regex=False
        )

    def split_sections(
        self,
        sections: List[ParsedSection],
        document_id: str,
        document_title: str,
        collection_name: str = "default",
        extra_metadata: Optional[Dict[str, Any]] = None
    ) -> List[LCDocument]:
        """
        Splits a list of parsed document sections into LangChain Document chunks.
        """
        lc_docs: List[LCDocument] = []
        chunk_counter = 0

        for section in sections:
            if not section.text.strip():
                continue

            # Split raw text of section
            sub_chunks = self._splitter.split_text(section.text)

            for sub_text in sub_chunks:
                cleaned_chunk = sub_text.strip()
                if not cleaned_chunk:
                    continue

                metadata: Dict[str, Any] = {
                    "doc_id": document_id,
                    "title": document_title,
                    "chunk_index": chunk_counter,
                    "collection": collection_name,
                    "page": section.page_number or 1,
                    "char_count": len(cleaned_chunk),
                    **(section.metadata or {}),
                    **(extra_metadata or {})
                }

                lc_docs.append(
                    LCDocument(
                        page_content=cleaned_chunk,
                        metadata=metadata
                    )
                )
                chunk_counter += 1

        return lc_docs

    def split_raw_text(
        self,
        text: str,
        document_id: str,
        document_title: str,
        collection_name: str = "default",
        extra_metadata: Optional[Dict[str, Any]] = None
    ) -> List[LCDocument]:
        """Convenience method to split a single string directly."""
        section = ParsedSection(text=text, page_number=1)
        return self.split_sections(
            sections=[section],
            document_id=document_id,
            document_title=document_title,
            collection_name=collection_name,
            extra_metadata=extra_metadata
        )


# Global singleton splitter
rag_text_splitter = RAGTextSplitter()
