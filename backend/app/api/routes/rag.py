"""
RAG & Document Ingestion Router (backend/app/api/routes/rag.py)
---------------------------------------------------------------
PYTHON CONCEPTS DEMONSTRATED:
- FastAPI Route Handlers: Multipart file upload, JSON payload validation,
  and path parameter extraction.
- Dependency Injection: Secures endpoints via `get_current_user` and database session `get_db`.
- Standardized API Envelope: Wraps responses in `APIResponse[T]` for consistent frontend consumption.
"""

from typing import List, Optional
from fastapi import (
    APIRouter,
    Depends,
    File,
    Form,
    HTTPException,
    Query,
    UploadFile,
    status
)
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.logging import logger
from app.api.deps import get_db, get_current_user
from app.models.user import User
from app.schemas.common import APIResponse
from app.schemas.rag import (
    DocumentResponse,
    RawDocumentIngestRequest,
    RAGQueryRequest,
    RAGQueryResponse,
    RAGStatsResponse,
    RetrievedChunkSchema
)
from app.services.rag_service import rag_service

router = APIRouter(prefix="/rag", tags=["Retrieval-Augmented Generation (RAG)"])


@router.post(
    "/documents",
    response_model=APIResponse[DocumentResponse],
    summary="Upload and index a document (PDF, TXT, MD, CSV)"
)
async def upload_document(
    file: UploadFile = File(..., description="Document file to parse and index"),
    title: Optional[str] = Form(None, description="Optional custom document title"),
    collection_name: str = Form("default", description="Knowledge collection partition"),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Parses an uploaded file, splits it into semantic chunks, generates dense vector
    embeddings, indices them in FAISS, and tracks the document in the database.
    """
    file_bytes = await file.read()
    if len(file_bytes) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file is empty."
        )

    try:
        doc = await rag_service.ingest_file(
            db=db,
            user_id=current_user.id,
            file_bytes=file_bytes,
            filename=file.filename or "uploaded_file.txt",
            title=title,
            collection_name=collection_name
        )
        return APIResponse(
            success=True,
            message=f"Document '{doc.title}' indexed successfully with {doc.chunk_count} chunks.",
            data=DocumentResponse.model_validate(doc)
        )
    except Exception as exc:
        logger.error(f"Document ingestion failed: {exc}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to parse and index document: {str(exc)}"
        )


@router.post(
    "/documents/raw",
    response_model=APIResponse[DocumentResponse],
    summary="Index raw text or markdown snippet"
)
async def ingest_raw_text(
    payload: RawDocumentIngestRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Directly chunks and indexes a raw text or markdown snippet into FAISS.
    """
    try:
        doc = await rag_service.ingest_raw_text(
            db=db,
            user_id=current_user.id,
            title=payload.title,
            content=payload.content,
            collection_name=payload.collection_name
        )
        return APIResponse(
            success=True,
            message=f"Snippet '{doc.title}' indexed successfully with {doc.chunk_count} chunks.",
            data=DocumentResponse.model_validate(doc)
        )
    except Exception as exc:
        logger.error(f"Raw snippet ingestion failed: {exc}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to index text snippet: {str(exc)}"
        )


@router.get(
    "/documents",
    response_model=APIResponse[List[DocumentResponse]],
    summary="List all indexed knowledge documents"
)
async def list_documents(
    collection_name: Optional[str] = Query(None, description="Filter by collection"),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Lists all knowledge documents uploaded by the authenticated user."""
    docs = await rag_service.list_documents(
        db=db,
        user_id=current_user.id,
        collection_name=collection_name
    )
    return APIResponse(
        success=True,
        data=[DocumentResponse.model_validate(d) for d in docs]
    )


@router.get(
    "/documents/{document_id}",
    response_model=APIResponse[DocumentResponse],
    summary="Get single document details"
)
async def get_document(
    document_id: str,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Retrieves document metadata by ID."""
    doc = await rag_service.get_document(db=db, document_id=document_id, user_id=current_user.id)
    if not doc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found."
        )
    return APIResponse(
        success=True,
        data=DocumentResponse.model_validate(doc)
    )


@router.delete(
    "/documents/{document_id}",
    response_model=APIResponse[bool],
    summary="Delete a knowledge document and its FAISS vector embeddings"
)
async def delete_document(
    document_id: str,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Deletes the document record, deletes the source file from disk,
    and removes all related chunk vectors from the FAISS index.
    """
    deleted = await rag_service.delete_document(
        db=db,
        document_id=document_id,
        user_id=current_user.id
    )
    if not deleted:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found."
        )
    return APIResponse(
        success=True,
        message="Document and vector embeddings deleted successfully.",
        data=True
    )


@router.post(
    "/query",
    response_model=APIResponse[RAGQueryResponse],
    summary="Semantic similarity query against knowledge base"
)
async def query_knowledge_base(
    payload: RAGQueryRequest,
    current_user: User = Depends(get_current_user)
):
    """
    Searches the FAISS vector index for the top-k most semantically relevant chunks,
    returning both the raw matches with similarity scores and an assembled context block.
    """
    try:
        retrieval = await rag_service.query_knowledge_base(
            user_id=current_user.id,
            query=payload.query,
            top_k=payload.top_k,
            score_threshold=payload.score_threshold,
            collection_name=payload.collection_name
        )

        chunk_schemas = [
            RetrievedChunkSchema(
                text=c.text,
                score=c.score,
                doc_id=c.doc_id,
                title=c.title,
                chunk_index=c.chunk_index,
                page=c.page,
                metadata=c.metadata
            )
            for c in retrieval.chunks
        ]

        response_payload = RAGQueryResponse(
            query=retrieval.query,
            chunks=chunk_schemas,
            context_text=retrieval.context_text,
            chunk_count=retrieval.chunk_count,
            top_score=retrieval.top_score,
            latency_ms=retrieval.latency_ms
        )

        return APIResponse(
            success=True,
            message=f"Retrieved {retrieval.chunk_count} relevant chunks in {retrieval.latency_ms} ms.",
            data=response_payload
        )
    except Exception as exc:
        logger.error(f"RAG query failed: {exc}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Semantic retrieval failed: {str(exc)}"
        )


@router.get(
    "/stats",
    response_model=APIResponse[RAGStatsResponse],
    summary="Get collection and vector store statistics"
)
async def get_rag_stats(
    collection_name: str = Query("default", description="Collection partition"),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Returns vector store size, total indexed chunks, and document count."""
    stats = await rag_service.get_stats(
        db=db,
        user_id=current_user.id,
        collection_name=collection_name
    )
    return APIResponse(
        success=True,
        data=RAGStatsResponse(
            collection_name=stats["collection_name"],
            chunk_count=stats["chunk_count"],
            size_bytes=stats["size_bytes"],
            document_count=stats["document_count"],
            exists=stats["exists"]
        )
    )
