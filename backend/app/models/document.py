"""
Knowledge Document Model (backend/app/models/document.py)
---------------------------------------------------------
PYTHON CONCEPTS DEMONSTRATED:
- Relational mapping of user uploaded documents for RAG (Retrieval-Augmented Generation).
- Metadata tracking for indexing state, chunk counts, and collection partitions.
"""

from sqlalchemy import Column, String, Integer, Boolean, ForeignKey
from sqlalchemy.orm import relationship
from app.core.database import Base
from app.models.base import BaseModelMixin


class Document(Base, BaseModelMixin):
    """
    Represents an uploaded knowledge base document (PDF, TXT, DOCX, URL)
    indexed in FAISS.
    """
    __tablename__ = "documents"

    user_id = Column(String(36), ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    title = Column(String(255), nullable=False)
    file_path = Column(String(500), nullable=True)
    file_type = Column(String(50), nullable=False)  # pdf, txt, docx, url
    file_size_bytes = Column(Integer, default=0, nullable=False)
    chunk_count = Column(Integer, default=0, nullable=False)
    collection_name = Column(String(100), default="default", nullable=False, index=True)
    is_indexed = Column(Boolean, default=False, nullable=False)

    # Relationships
    user = relationship("User", back_populates="documents")

    def __repr__(self) -> str:
        return f"<Document id={self.id} title={self.title} indexed={self.is_indexed}>"
