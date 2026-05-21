"""
app.py — FastAPI server for the fine-tuned Python code assistant
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
import time
import os

app = FastAPI(
    title="LLM Code Finetuner API",
    description="Fine-tuned Mistral-7B Python code assistant",
    version="1.0.0",
)

MODEL_NAME   = "mistralai/Mistral-7B-v0.1"
ADAPTER_PATH = os.environ.get("ADAPTER_PATH", "./lora-adapter")

model     = None
tokenizer = None

class CodeRequest(BaseModel):
    instruction: str
    input: str = ""
    max_new_tokens: int = 256

class CodeResponse(BaseModel):
    instruction: str
    generated_code: str
    latency_ms: float

def load_model():
    global model, tokenizer
    print("Loading model...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, quantization_config=bnb_config, device_map="auto"
    )
    model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
    print("Model ready!")

@app.on_event("startup")
async def startup_event():
    load_model()

@app.get("/")
def root():
    return {"status": "ok", "model": "mistral-7b-qlora-finetuned"}

@app.get("/health")
def health():
    return {"status": "healthy", "model_loaded": model is not None}

@app.post("/generate", response_model=CodeResponse)
def generate_code(request: CodeRequest):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    prompt = f"""### Instruction:
{request.instruction}

### Input:
{request.input}

### Response:
"""
    start = time.time()
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=request.max_new_tokens,
            temperature=0.1,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    generated = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
    )
    latency_ms = (time.time() - start) * 1000

    return CodeResponse(
        instruction=request.instruction,
        generated_code=generated.strip(),
        latency_ms=round(latency_ms, 2),
    )
