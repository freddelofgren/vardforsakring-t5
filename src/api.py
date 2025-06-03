# src/api.py

from fastapi import FastAPI, Body, HTTPException
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import json

app = FastAPI(
    title="Vårdförsäkring T5-API",
    description="Analyserar en hel försäkringstext (även mycket långa) "
                "genom att dela upp i chunkar, köra ‘extract’ per chunk och "
                "sedan ‘compare’ + ‘faq’ på det aggregerade resultatet.",
    version="1.0.0"
)

# ----------------------------------------------------------------------------------
# Initiera modell och tokenizer
# ----------------------------------------------------------------------------------
MODEL_PATH = "models/all-tasks-t5-swedish"
tokenizer  = AutoTokenizer.from_pretrained(MODEL_PATH)
model      = AutoModelForSeq2SeqLM.from_pretrained(MODEL_PATH)
device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

def chunk_text_by_tokens(text: str, max_tokens: int = 1000) -> list[str]:
    """
    Dela upp texten i mindre strängar, vardera max max_tokens token långa.
    """
    encoding = tokenizer(text, return_offsets_mapping=True, add_special_tokens=False)
    offsets = encoding["offset_mapping"]
    total_tokens = len(offsets)
    chunks = []
    start = 0
    while start < total_tokens:
        end = min(start + max_tokens, total_tokens)
        char_start = offsets[start][0]
        char_end   = offsets[end - 1][1]
        chunks.append(text[char_start:char_end])
        start = end
    return chunks

def extract_chunk(chunk_text: str) -> dict:
    """
    Kör 'extract' på en enda chunk och returnerar ett dict (eller tomt om fel).
    """
    prompt = f"extract: {chunk_text}"
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=1024
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    outputs = model.generate(
        **inputs,
        max_length=512,
        num_beams=4,
        early_stopping=True
    )
    out_str = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
    try:
        return json.loads(out_str)
    except (json.JSONDecodeError, TypeError):
        return {}

def aggregate_extract(chunks: list[str]) -> dict:
    """
    Går igenom alla chunkar, kör extract på varje, och OR-aggregerar boolean-fälten.
    """
    aggregated = {}
    for i, chunk in enumerate(chunks):
        data = extract_chunk(chunk)
        if not data:
            continue
        if i == 0:
            aggregated = data.copy()
        else:
            for key, val in data.items():
                if key in aggregated and isinstance(val, bool):
                    aggregated[key] = aggregated[key] or val
                # övriga fält (t.ex. "försäkring") behåller vi från första chunk
    return aggregated

@app.post(
    "/analyze",
    summary="Analysera en lång försäkringstext",
    description=(
        "Tar emot _ren text_ (Content-Type: text/plain), även >100 000 tecken. "
        "Delas upp i token‐baserade chunkar. Kör 'extract' på varje chunk, aggregerar "
        "boolean‐fälten, och kör därefter 'compare' och 'faq' på det aggregerade resultatet."
    )
)
async def analyze_long(
    text: str = Body(
        ...,
        media_type="text/plain",
        description="Hela försäkringstexten (ingen JSON‐inpackning), "
                    "kan vara mycket lång (>100 000 tecken)."
    )
):
    # 1) Dela upp texten i chunkar om max 1 000 token.
    chunks = chunk_text_by_tokens(text, max_tokens=1000)
    if not chunks:
        raise HTTPException(status_code=400, detail="Tom input‐text eller kunde ej chunkas.")

    # 2) Kör extract per chunk och aggregera boolean‐fält
    extract_agg = aggregate_extract(chunks)
    if not extract_agg:
        raise HTTPException(status_code=500, detail="Extract‐delen gav inget giltigt resultat på någon chunk.")

    # 3) Kör compare på det aggregerade extract‐resultatet
    compare_input_json = json.dumps(extract_agg, ensure_ascii=False)
    compare_prompt     = f"compare: {compare_input_json}"
    inputs = tokenizer(
        compare_prompt,
        return_tensors="pt",
        truncation=True,
        max_length=512
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    compare_outputs = model.generate(
        **inputs,
        max_length=512,
        num_beams=4,
        early_stopping=True
    )
    compare_json_str = tokenizer.batch_decode(compare_outputs, skip_special_tokens=True)[0]
    try:
        compare_data = json.loads(compare_json_str)
    except (json.JSONDecodeError, TypeError):
        compare_data = {"compare_raw": compare_json_str}

    # 4) Kör faq på samma aggregerade extract‐data
    faq_prompt = f"faq: {compare_input_json}"
    inputs = tokenizer(
        faq_prompt,
        return_tensors="pt",
        truncation=True,
        max_length=512
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    faq_outputs = model.generate(
        **inputs,
        max_length=1024,
        num_beams=4,
        early_stopping=True
    )
    faq_json_str = tokenizer.batch_decode(faq_outputs, skip_special_tokens=True)[0]
    try:
        faq_data = json.loads(faq_json_str)
    except (json.JSONDecodeError, TypeError):
        faq_data = {"faq_raw": faq_json_str}

    # 5) Returnera allt som JSON
    return {
        "extract": extract_agg,
        "compare": compare_data,
        "faq": faq_data
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
