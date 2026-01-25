"""FastAPI app placeholder."""

from fastapi import FastAPI

app = FastAPI(title="Biotech Trial Forecasting API")


@app.get("/health")
async def health():
    return {"status": "ok"}
