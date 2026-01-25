"""FastAPI app placeholder."""

from fastapi import FastAPI

app = FastAPI(title="ai-clinical-trials-rag API")


@app.get("/health")
async def health():
    return {"status": "ok"}
