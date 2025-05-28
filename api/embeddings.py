from fastapi import APIRouter, HTTPException, Depends, status
from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime
from sqlalchemy.orm import Session
from models.models import Embedding
from connect_db import get_db
from firebase_auth import verify_firebase_token
from utils.logger import logger
import uuid

router = APIRouter()

# Pydantic models
class EmbeddingCreate(BaseModel):
    reminder_id: Optional[str] = None
    embedding: List[float]

class EmbeddingResponse(BaseModel):
    id: str
    user_id: str
    reminder_id: Optional[str]
    embedding: List[float]
    created_at: datetime

    class Config:
        from_attributes = True

class SimilaritySearchRequest(BaseModel):
    query_embedding: List[float]
    limit: Optional[int] = 10
    threshold: Optional[float] = 0.8

class SimilarityResult(BaseModel):
    embedding: EmbeddingResponse
    similarity_score: float

@router.post("/", response_model=EmbeddingResponse)
async def create_embedding(
    embedding_data: EmbeddingCreate,
    current_user: dict = Depends(verify_firebase_token),
    db: Session = Depends(get_db)
):
    """Create a new embedding."""
    try:
        user_id = current_user["uid"]
        
        embedding = Embedding(
            id=str(uuid.uuid4()),
            user_id=user_id,
            reminder_id=embedding_data.reminder_id,
            embedding=embedding_data.embedding
        )
        
        db.add(embedding)
        db.commit()
        db.refresh(embedding)
        
        logger.info(f"Created embedding: {embedding.id} for user: {user_id}")
        return EmbeddingResponse(
            id=str(embedding.id),
            user_id=str(embedding.user_id),
            reminder_id=str(embedding.reminder_id) if embedding.reminder_id else None,
            embedding=embedding.embedding,
            created_at=embedding.created_at
        )
        
    except Exception as e:
        db.rollback()
        logger.error(f"Create embedding error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

@router.get("/", response_model=List[EmbeddingResponse])
async def get_embeddings(
    reminder_id: Optional[str] = None,
    current_user: dict = Depends(verify_firebase_token),
    db: Session = Depends(get_db)
):
    """Get all embeddings for the current user."""
    try:
        user_id = current_user["uid"]
        query = db.query(Embedding).filter(Embedding.user_id == user_id)
        
        if reminder_id:
            query = query.filter(Embedding.reminder_id == reminder_id)
        
        embeddings = query.order_by(Embedding.created_at.desc()).all()
        
        return [
            EmbeddingResponse(
                id=str(embedding.id),
                user_id=str(embedding.user_id),
                reminder_id=str(embedding.reminder_id) if embedding.reminder_id else None,
                embedding=embedding.embedding,
                created_at=embedding.created_at
            )
            for embedding in embeddings
        ]
        
    except Exception as e:
        logger.error(f"Get embeddings error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

@router.get("/{embedding_id}", response_model=EmbeddingResponse)
async def get_embedding(
    embedding_id: str,
    current_user: dict = Depends(verify_firebase_token),
    db: Session = Depends(get_db)
):
    """Get a specific embedding."""
    try:
        user_id = current_user["uid"]
        embedding = db.query(Embedding).filter(
            Embedding.id == embedding_id,
            Embedding.user_id == user_id
        ).first()
        
        if not embedding:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Embedding not found"
            )
        
        return EmbeddingResponse(
            id=str(embedding.id),
            user_id=str(embedding.user_id),
            reminder_id=str(embedding.reminder_id) if embedding.reminder_id else None,
            embedding=embedding.embedding,
            created_at=embedding.created_at
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Get embedding error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

@router.delete("/{embedding_id}")
async def delete_embedding(
    embedding_id: str,
    current_user: dict = Depends(verify_firebase_token),
    db: Session = Depends(get_db)
):
    """Delete an embedding."""
    try:
        user_id = current_user["uid"]
        embedding = db.query(Embedding).filter(
            Embedding.id == embedding_id,
            Embedding.user_id == user_id
        ).first()
        
        if not embedding:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Embedding not found"
            )
        
        db.delete(embedding)
        db.commit()
        
        logger.info(f"Deleted embedding: {embedding_id}")
        return {"message": "Embedding deleted successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        logger.error(f"Delete embedding error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

@router.post("/search", response_model=List[SimilarityResult])
async def search_similar_embeddings(
    search_request: SimilaritySearchRequest,
    current_user: dict = Depends(verify_firebase_token),
    db: Session = Depends(get_db)
):
    """Search for similar embeddings using cosine similarity."""
    try:
        user_id = current_user["uid"]
        
        # Get all embeddings for the user
        embeddings = db.query(Embedding).filter(Embedding.user_id == user_id).all()
        
        # Calculate cosine similarity for each embedding
        def cosine_similarity(vec1, vec2):
            """Calculate cosine similarity between two vectors."""
            import numpy as np
            vec1, vec2 = np.array(vec1), np.array(vec2)
            dot_product = np.dot(vec1, vec2)
            norm_product = np.linalg.norm(vec1) * np.linalg.norm(vec2)
            if norm_product == 0:
                return 0
            return dot_product / norm_product
        
        results = []
        for embedding in embeddings:
            similarity = cosine_similarity(search_request.query_embedding, embedding.embedding)
            
            if similarity >= search_request.threshold:
                results.append(SimilarityResult(
                    embedding=EmbeddingResponse(
                        id=str(embedding.id),
                        user_id=str(embedding.user_id),
                        reminder_id=str(embedding.reminder_id) if embedding.reminder_id else None,
                        embedding=embedding.embedding,
                        created_at=embedding.created_at
                    ),
                    similarity_score=float(similarity)
                ))
        
        # Sort by similarity score in descending order
        results.sort(key=lambda x: x.similarity_score, reverse=True)
        
        # Limit results
        return results[:search_request.limit]
        
    except Exception as e:
        logger.error(f"Similarity search error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        ) 