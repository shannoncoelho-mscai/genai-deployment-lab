
import os
import time
from contextlib import asynccontextmanager

import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "HuggingFaceTB/SmolLM2-360M-Instruct"
MAX_NEW_TOKENS = 128
TEMPERATURE = 0.7

tokenizer = None
model = None
model_load_seconds = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global tokenizer, model, model_load_seconds

    start = time.perf_counter()
    print(f"Loading model: {MODEL_ID}", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float32,
    )
    model.eval()

    model_load_seconds = time.perf_counter() - start
    print(f"Model loaded in {model_load_seconds:.2f} seconds", flush=True)

    yield


app = FastAPI(
    title="Open-Weight GenAI API",
    description="CPU-based SmolLM2 text generation",
    lifespan=lifespan,
)


class PromptRequest(BaseModel):
    prompt: str = Field(min_length=1, max_length=2000)


@app.get("/")
def root():
    return {
        "message": "SmolLM2 GenAI API",
        "model": MODEL_ID,
        "max_new_tokens": MAX_NEW_TOKENS,
        "temperature": TEMPERATURE,
    }


@app.get("/health")
def health():
    return {
        "status": "healthy" if model is not None else "loading",
        "model_loaded": model is not None,
        "model_load_seconds": model_load_seconds,
    }


@app.post("/generate")
def generate(request: PromptRequest):
    if model is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="Model is not ready")

    start = time.perf_counter()

    messages = [{"role": "user", "content": request.prompt}]
    formatted_prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = tokenizer(formatted_prompt, return_tensors="pt")

    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=True,
            temperature=TEMPERATURE,
            pad_token_id=tokenizer.eos_token_id,
        )

    generated_tokens = output[0][inputs["input_ids"].shape[1]:]
    response_text = tokenizer.decode(
        generated_tokens,
        skip_special_tokens=True,
    )

    elapsed = time.perf_counter() - start

    return {
        "model": MODEL_ID,
        "response": response_text,
        "response_time_seconds": round(elapsed, 3),
        "output_tokens": len(generated_tokens),
        "max_new_tokens": MAX_NEW_TOKENS,
        "temperature": TEMPERATURE,
    }
