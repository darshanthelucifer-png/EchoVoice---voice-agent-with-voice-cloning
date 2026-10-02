"""
Document Parser Module (backend/app/ai/rag/parser.py)
----------------------------------------------------
PYTHON CONCEPTS DEMONSTRATED:
- Multi-Format Ingestion: Robustly extracts text and structural metadata across PDFs,
  plain text, Markdown, CSV, and raw code files.
- Resilient Encoding Detection: Handles diverse text encodings (UTF-8, Latin-1, CP1252)
  without raising unhandled UnicodeDecodeErrors.
- In-Memory & File-Based Parsing: Supports streaming byte buffers and disk paths.
"""

from dataclasses import dataclass, field
import io
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import pypdf

from app.core.logging import logger


@dataclass
class ParsedSection:
    """Represents a discrete structural section or page extracted from a document."""
    text: str
    page_number: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class DocumentParser:
    """
    Extracts plain text and page metadata from multi-format knowledge files.
    """

    @staticmethod
    def parse_pdf(file_source: Union[str, Path, bytes, io.BytesIO]) -> List[ParsedSection]:
        """Extracts text page-by-page from a PDF document."""
        sections: List[ParsedSection] = []

        if isinstance(file_source, (str, Path)):
            stream = open(file_source, "rb")
        elif isinstance(file_source, bytes):
            stream = io.BytesIO(file_source)
        else:
            stream = file_source

        try:
            reader = pypdf.PdfReader(stream)
            for page_idx, page in enumerate(reader.pages, start=1):
                page_text = page.extract_text() or ""
                cleaned = page_text.strip()
                if cleaned:
                    sections.append(
                        ParsedSection(
                            text=cleaned,
                            page_number=page_idx,
                            metadata={"page": page_idx, "total_pages": len(reader.pages)}
                        )
                    )
        finally:
            if isinstance(file_source, (str, Path)):
                stream.close()

        return sections

    @staticmethod
    def parse_text(file_source: Union[str, Path, bytes], filename: str = "document.txt") -> List[ParsedSection]:
        """Decodes text files across standard character encodings."""
        raw_bytes: bytes
        if isinstance(file_source, (str, Path)):
            with open(file_source, "rb") as f:
                raw_bytes = f.read()
        else:
            raw_bytes = file_source

        text = ""
        for encoding in ["utf-8", "utf-8-sig", "latin-1", "cp1252"]:
            try:
                text = raw_bytes.decode(encoding)
                break
            except UnicodeDecodeError:
                continue

        cleaned = text.strip()
        if not cleaned:
            return []

        return [
            ParsedSection(
                text=cleaned,
                page_number=1,
                metadata={"filename": filename}
            )
        ]

    @classmethod
    def parse_file(
        cls,
        file_source: Union[str, Path, bytes],
        filename: str
    ) -> List[ParsedSection]:
        """Dispatches parser based on file extension."""
        ext = Path(filename).suffix.lower()

        if ext == ".pdf":
            return cls.parse_pdf(file_source)
        elif ext in [".txt", ".md", ".markdown", ".csv", ".json", ".log", ".py", ".html"]:
            return cls.parse_text(file_source, filename=filename)
        else:
            # Fallback text decoder
            logger.info(f"Unrecognized extension '{ext}', attempting generic text decode for '{filename}'")
            return cls.parse_text(file_source, filename=filename)


# Global singleton parser
document_parser = DocumentParser()
