"""
RAG Knowledge Base Service (backend/app/services/rag_service.py)
---------------------------------------------------------------
PYTHON CONCEPTS DEMONSTRATED:
- Service Facade Pattern: Orchestrates file persistence, parser dispatching,
  recursive text chunking, vector indexing in FAISS, and relational DB tracking.
- Transactional Consistency: Commits database state only after vector index serialization
  succeeds; cleans up disk artifacts on failure.
- Multi-Tenant Security: Scopes all document queries and vector stores to authenticated `user_id`.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional
import uuid
import aiofiles
from sqlalchemy import select, delete, func
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.core.logging import logger, timed_step
from app.models.document import Document
from app.ai.rag.parser import document_parser
from app.ai.rag.splitter import rag_text_splitter
from app.ai.rag.vector_store import vector_store_manager
from app.ai.rag.retriever import rag_retriever, RetrievalResult


class RAGService:
    """
    Coordinates end-to-end document ingestion, FAISS vector indexing,
    and semantic context retrieval for the EchoVoice voice assistant.
    """

    @timed_step("Ingest Knowledge File")
    async def ingest_file(
        self,
        db: AsyncSession,
        user_id: str,
        file_bytes: bytes,
        filename: str,
        title: Optional[str] = None,
        collection_name: str = "default",
        use_fallback: bool = False
    ) -> Document:
        """
        Ingests a binary file (PDF, TXT, MD, etc.), saves to disk,
        extracts text, chunks, embeds into FAISS, and registers in the DB.
        """
        doc_id = str(uuid.uuid4())
        doc_title = title or Path(filename).stem.replace("_", " ").replace("-", " ").title()
        file_ext = Path(filename).suffix.lower().lstrip(".") or "txt"

        # 1. Save file to disk
        user_docs_dir = settings.DOCUMENTS_DIR / user_id
        user_docs_dir.mkdir(parents=True, exist_ok=True)
        file_path = user_docs_dir / f"{doc_id}_{filename}"

        async with aiofiles.open(file_path, "wb") as f:
            await f.write(file_bytes)

        try:
            # 2. Parse structural text sections
            sections = document_parser.parse_file(file_source=file_bytes, filename=filename)
            if not sections:
                raise ValueError(f"No extractable text found in '{filename}'.")

            # 3. Chunk text recursively
            chunks = rag_text_splitter.split_sections(
                sections=sections,
                document_id=doc_id,
                document_title=doc_title,
                collection_name=collection_name,
                extra_metadata={"filename": filename, "user_id": user_id}
            )

            # 4. Add to FAISS Vector Store
            chunk_count = vector_store_manager.add_documents(
                documents=chunks,
                user_id=user_id,
                collection_name=collection_name,
                use_fallback=use_fallback
            )

            # 5. Persist Document record in DB
            db_doc = Document(
                id=doc_id,
                user_id=user_id,
                title=doc_title,
                file_path=str(file_path),
                file_type=file_ext,
                file_size_bytes=len(file_bytes),
                chunk_count=chunk_count,
                collection_name=collection_name,
                is_indexed=True
            )
            db.add(db_doc)
            await db.commit()
            await db.refresh(db_doc)

            logger.info(
                f"Successfully ingested document '{doc_title}' ({doc_id}) "
                f"into collection '{collection_name}' with {chunk_count} chunks."
            )
            return db_doc

        except Exception as exc:
            # Clean up saved file on failure
            if file_path.exists():
                file_path.unlink(missing_ok=True)
            logger.error(f"Failed to ingest document '{filename}': {exc}", exc_info=True)
            raise

    @timed_step("Ingest Raw Text Document")
    async def ingest_raw_text(
        self,
        db: AsyncSession,
        user_id: str,
        title: str,
        content: str,
        collection_name: str = "default",
        use_fallback: bool = False
    ) -> Document:
        """
        Directly ingests a raw string or markdown text snippet.
        """
        doc_id = str(uuid.uuid4())
        user_docs_dir = settings.DOCUMENTS_DIR / user_id
        user_docs_dir.mkdir(parents=True, exist_ok=True)
        file_path = user_docs_dir / f"{doc_id}_content.txt"

        async with aiofiles.open(file_path, "w", encoding="utf-8") as f:
            await f.write(content)

        try:
            chunks = rag_text_splitter.split_raw_text(
                text=content,
                document_id=doc_id,
                document_title=title,
                collection_name=collection_name,
                extra_metadata={"user_id": user_id}
            )

            chunk_count = vector_store_manager.add_documents(
                documents=chunks,
                user_id=user_id,
                collection_name=collection_name,
                use_fallback=use_fallback
            )

            db_doc = Document(
                id=doc_id,
                user_id=user_id,
                title=title,
                file_path=str(file_path),
                file_type="txt",
                file_size_bytes=len(content.encode("utf-8")),
                chunk_count=chunk_count,
                collection_name=collection_name,
                is_indexed=True
            )
            db.add(db_doc)
            await db.commit()
            await db.refresh(db_doc)

            return db_doc
        except Exception as exc:
            if file_path.exists():
                file_path.unlink(missing_ok=True)
            logger.error(f"Failed to ingest raw text '{title}': {exc}", exc_info=True)
            raise

    async def list_documents(
        self,
        db: AsyncSession,
        user_id: str,
        collection_name: Optional[str] = None
    ) -> List[Document]:
        """Lists knowledge documents belonging to the user."""
        query = select(Document).where(Document.user_id == user_id)
        if collection_name:
            query = query.where(Document.collection_name == collection_name)
        query = query.order_by(Document.created_at.desc())
        result = await db.execute(query)
        return list(result.scalars().all())

    async def get_document(
        self,
        db: AsyncSession,
        document_id: str,
        user_id: str
    ) -> Optional[Document]:
        """Retrieves a single document by ID scoped to user."""
        query = select(Document).where(
            Document.id == document_id,
            Document.user_id == user_id
        )
        result = await db.execute(query)
        return result.scalar_one_or_none()

    @timed_step("Delete Knowledge Document")
    async def delete_document(
        self,
        db: AsyncSession,
        document_id: str,
        user_id: str,
        use_fallback: bool = False
    ) -> bool:
        """
        Deletes vector chunks from FAISS, disk file, and database record.
        """
        doc = await self.get_document(db, document_id, user_id)
        if not doc:
            return False

        # 1. Remove vector chunks from FAISS
        vector_store_manager.delete_document_chunks(
            document_id=document_id,
            user_id=user_id,
            collection_name=doc.collection_name,
            use_fallback=use_fallback
        )

        # 2. Delete source file on disk
        if doc.file_path and Path(doc.file_path).exists():
            Path(doc.file_path).unlink(missing_ok=True)

        # 3. Delete DB record
        await db.delete(doc)
        await db.commit()

        logger.info(f"Deleted document '{doc.title}' ({document_id}) from knowledge base.")
        return True

    async def query_knowledge_base(
        self,
        user_id: str,
        query: str,
        top_k: int = 4,
        score_threshold: Optional[float] = None,
        collection_name: str = "default",
        use_fallback: bool = False
    ) -> RetrievalResult:
        """
        Retrieves relevant context chunks and formatted prompt block.
        """
        return await rag_retriever.retrieve(
            query=query,
            k=top_k,
            score_threshold=score_threshold,
            user_id=user_id,
            collection_name=collection_name,
            use_fallback=use_fallback
        )

    async def get_stats(
        self,
        db: AsyncSession,
        user_id: str,
        collection_name: str = "default"
    ) -> Dict[str, Any]:
        """Returns aggregated database and vector store telemetry."""
        # Query document count from DB
        stmt = select(func.count(Document.id)).where(
            Document.user_id == user_id,
            Document.collection_name == collection_name
        )
        result = await db.execute(stmt)
        doc_count = result.scalar_one() or 0

        # Query vector store stats
        vs_stats = vector_store_manager.get_collection_stats(
            user_id=user_id,
            collection_name=collection_name
        )

        return {
            "collection_name": collection_name,
            "document_count": doc_count,
            "chunk_count": vs_stats["chunk_count"],
            "size_bytes": vs_stats["size_bytes"],
            "exists": vs_stats["exists"]
        }


# Global singleton service
rag_service = RAGService()
