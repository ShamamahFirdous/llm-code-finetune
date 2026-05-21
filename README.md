# LLM Code Finetuner

Fine-tuning Mistral-7B on Python code generation using QLoRA — achieving significant improvement over the base model with only 0.094% of parameters trained.

## Overview

This project fine-tunes `mistralai/Mistral-7B-v0.1` on 2,000 Python coding instruction pairs using QLoRA (Quantized Low-Rank Adaptation). The fine-tuned model is evaluated against the base model, tracked with MLflow, and served via a FastAPI endpoint containerized with Docker.

## Architecture

```
Dataset (18K Python instructions)
        │
        ▼
┌─────────────────────┐
│  Data Formatting    │  Instruction / Input / Response template
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  Mistral-7B (4-bit) │  BitsAndBytes NF4 quantization
│  + LoRA Adapter     │  r=16, alpha=32, target: q_proj, v_proj
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  SFTTrainer         │  Supervised Fine-Tuning
│  MLflow Tracking    │  Logs params, metrics, artifacts
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  EVALUATION         │  Fine-tuned vs base model comparison
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  FastAPI + Docker   │  Production-ready REST API
└─────────────────────┘
```

## Training Results

| Metric | Value |
|--------|-------|
| Base model | mistralai/Mistral-7B-v0.1 |
| Quantization | 4-bit NF4 (QLoRA) |
| Trainable parameters | 6,815,744 / 7,248,547,840 (0.094%) |
| Training samples | 2,000 |
| Epochs | 1 |
| Learning rate | 2e-4 |
| Initial loss | 0.156 |
| Final loss | ~0.0001 |

## Loss Curve

| Step | Training Loss |
|------|--------------|
| 10 | 0.156090 |
| 20 | 0.020436 |
| 30 | 0.003363 |
| 40 | 0.000321 |
| 50 | 0.000658 |
| 60 | 0.000412 |
| 70 | 0.000168 |
| 80 | 0.000864 |
| 90 | 0.000090 |
| 100 | 0.000075 |

## Tech Stack

- **Fine-tuning**: QLoRA via Hugging Face PEFT
- **Base model**: Mistral-7B-v0.1
- **Training**: TRL SFTTrainer
- **Experiment tracking**: MLflow
- **Serving**: FastAPI + Uvicorn
- **Containerization**: Docker
- **Dataset**: `iamtarun/python_code_instructions_18k_alpaca`

## Project Structure

```
llm-code-finetuner/
├── train.py          # QLoRA fine-tuning script
├── evaluate.py       # Model evaluation and comparison
├── app.py            # FastAPI REST API
├── Dockerfile        # Container definition
├── requirements.txt  # Python dependencies
└── README.md
```

## Setup & Usage

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Fine-tune the model
```bash
HF_TOKEN=your_token python train.py
```

### 3. Evaluate fine-tuned vs base model
```bash
python evaluate.py
```

### 4. Run the API
```bash
uvicorn app:app --host 0.0.0.0 --port 8000
```

### 5. Run with Docker
```bash
docker build -t llm-code-finetuner .
docker run -p 8000:8000 llm-code-finetuner
```

### 6. Example API call
```bash
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{
    "instruction": "Write a Python function to check if a number is prime",
    "input": "number = 17"
  }'
```

## Key Technical Decisions

**Why QLoRA?** Training only 0.094% of parameters (6.8M out of 7.2B) makes fine-tuning feasible on a single GPU while maintaining model quality.

**Why 4-bit NF4 quantization?** Reduces memory from ~14GB to ~4GB, enabling training on a T4 GPU.

**Why target q_proj and v_proj?** These attention projection layers capture the most task-specific behavior with minimal parameter overhead.

## MLflow Experiment Tracking

All training runs are logged with MLflow including hyperparameters, loss curves, and evaluation metrics enabling reproducible experimentation.
