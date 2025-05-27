# src/api.py
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline

app = FastAPI()

# 1. Ladda extract_relevant‐modellen
extract_pipe = pipeline(
    "text2text-generation",
    model="models/extract-relevant",
    tokenizer="models/extract-relevant"
)

# 2. Ladda FAQ‐modellen
faq_pipe = pipeline(
    "text2text-generation",
    model="models/t5-faq",
    tokenizer="models/t5-faq"
)

# 3. Ladda jämförelse‐modellen
feat_pipe = pipeline(
    "text2text-generation",
    model="models/t5-jamforelse",
    tokenizer="models/t5-jamforelse"
)

class AnalyzeRequest(BaseModel):
    text: str

@app.post("/analyze/")
def analyze(req: AnalyzeRequest):
    # Steg 1: extrahera relevanta stycken
    out = extract_pipe(f"extract_relevant: {req.text}", max_length=300)[0]
    snippet = out["generated_text"]

    # Steg 2: FAQ‐svar
    faq_answer = faq_pipe(f"qa: {req.text} {snippet}", max_length=128)[0]["generated_text"]

    # Steg 3: feature‐JSON
    feat_output = feat_pipe(f"extract_features: {snippet}", max_length=512)[0]["generated_text"]

    return {
        "snippet": snippet,
        "faq_answer": faq_answer,
        "features": feat_output
    }
