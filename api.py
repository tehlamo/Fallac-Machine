from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os

from service.analyzer import analyze_text

app = FastAPI(title="Fallacy Detector API")


class AnalyzeRequest(BaseModel):
	text: str
	model_id: str | None = None
	threshold: float = 0.6


class AnalyzeResponse(BaseModel):
	input_text: str
	elapsed_seconds: float
	fallacies: list
	fallacy_types: list[str]
	sentences_with_fallacies: list[str]


@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze(req: AnalyzeRequest):
	model_id = req.model_id or os.getenv("FALLACY_MODEL_ID")
	if not model_id:
		raise HTTPException(status_code=400, detail="Model ID not provided (set model_id or FALLACY_MODEL_ID)")
	try:
		result = analyze_text(req.text, model_id=model_id, threshold=req.threshold)
		return AnalyzeResponse(**result)
	except Exception as e:
		raise HTTPException(status_code=500, detail=str(e))
