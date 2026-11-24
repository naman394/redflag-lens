"""FastAPI service.

  uvicorn src.app.api:app --port 8000
  curl -F file=@data/raw/inbox/GENSOL_FY2024.pdf -F company=GENSOL localhost:8000/analyze
"""
import json
from datetime import datetime

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from pydantic import BaseModel

from src.app.pipeline import LOGS, ReportError, analyze, classifier

app = FastAPI(title="RedFlag Lens", version="1.0")
MAX_BYTES = 60 * 1024 * 1024


@app.on_event("startup")
def load_models():
    classifier()  # load once, not per request


@app.get("/health")
def health():
    c = classifier()
    return {"status": "ok", "model": c.name, "threshold": c.threshold}


@app.post("/analyze")
async def analyze_pdf(file: UploadFile = File(...), company: str = Form("Uploaded report"), fy: str = Form(""),
                      use_llm: bool = Form(True)):
    data = await file.read()
    if len(data) > MAX_BYTES:
        raise HTTPException(413, f"File is larger than {MAX_BYTES // 2**20} MB.")
    try:
        return analyze(data, company=company, fy=fy, use_llm=use_llm)
    except ReportError as e:
        raise HTTPException(422, str(e))


class Feedback(BaseModel):
    company: str
    finding_text: str
    page: int
    vote: int  # +1 correct red flag, -1 wrong
    comment: str = ""


@app.post("/feedback")
def feedback(fb: Feedback):
    LOGS.mkdir(exist_ok=True)
    with open(LOGS / "feedback.jsonl", "a") as f:
        f.write(json.dumps({"ts": datetime.now().isoformat(timespec="seconds"), **fb.model_dump()}) + "\n")
    return {"saved": True}
