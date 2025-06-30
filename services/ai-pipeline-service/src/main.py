from fastapi import FastAPI
from .routers import pipeline

app = FastAPI(title="AI Pipeline Service", version="1.0.0")

app.include_router(pipeline.router, prefix="/pipeline", tags=["pipeline"])

@app.get("/health")
async def health():
    return {"status": "healthy", "service": "ai-pipeline-service"} 
 
 