"""
train.py — Fine-tune Mistral-7B on Python code instructions using QLoRA
"""

import torch
import mlflow
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer
from huggingface_hub import login
import os

# ── Config ──────────────────────────────────────────────────────────────────
HF_TOKEN       = os.environ.get("HF_TOKEN", "")
MODEL_NAME     = "mistralai/Mistral-7B-v0.1"
DATASET_NAME   = "iamtarun/python_code_instructions_18k_alpaca"
NUM_SAMPLES    = 2000
OUTPUT_DIR     = "./results"
LORA_R         = 16
LORA_ALPHA     = 32
LORA_DROPOUT   = 0.05
LEARNING_RATE  = 2e-4
EPOCHS         = 1
BATCH_SIZE     = 2
GRAD_ACC_STEPS = 4

# ── Hugging Face login ───────────────────────────────────────────────────────
login(token=HF_TOKEN)

# ── MLflow ───────────────────────────────────────────────────────────────────
mlflow.set_experiment("llm-code-finetuner")
with mlflow.start_run(run_name="mistral-7b-qlora"):

    mlflow.log_params({
        "model":            MODEL_NAME,
        "dataset":          DATASET_NAME,
        "num_samples":      NUM_SAMPLES,
        "lora_r":           LORA_R,
        "lora_alpha":       LORA_ALPHA,
        "lora_dropout":     LORA_DROPOUT,
        "learning_rate":    LEARNING_RATE,
        "epochs":           EPOCHS,
        "quantization":     "4-bit NF4",
        "target_modules":   "q_proj,v_proj",
    })

    # ── Dataset ──────────────────────────────────────────────────────────────
    print("Loading dataset...")
    dataset = load_dataset(DATASET_NAME)

    def format_prompt(example):
        return {"completion": f"""### Instruction:
{example['instruction']}

### Input:
{example['input']}

### Response:
{example['output']}"""}

    dataset   = dataset.map(format_prompt)
    train_data = dataset["train"].select(range(NUM_SAMPLES))
    print(f"Dataset ready: {len(train_data)} examples")

    # ── Model ─────────────────────────────────────────────────────────────────
    print("Loading model in 4-bit...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, quantization_config=bnb_config, device_map="auto"
    )
    print("Model loaded!")

    # ── LoRA ─────────────────────────────────────────────────────────────────
    model = prepare_model_for_kbit_training(model)
    lora_config = LoraConfig(
        r=LORA_R, lora_alpha=LORA_ALPHA,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=LORA_DROPOUT, bias="none", task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # ── Training ──────────────────────────────────────────────────────────────
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACC_STEPS,
        warmup_steps=20,
        learning_rate=LEARNING_RATE,
        bf16=True,
        logging_steps=10,
        save_steps=50,
        eval_strategy="no",
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=train_data,
        args=training_args,
        processing_class=tokenizer,
    )

    print("Starting training...")
    train_result = trainer.train()
    print("Training complete!")

    # ── Log metrics & save ───────────────────────────────────────────────────
    mlflow.log_metric("train_loss",        train_result.training_loss)
    mlflow.log_metric("train_runtime_sec", train_result.metrics["train_runtime"])

    model.save_pretrained("./lora-adapter")
    tokenizer.save_pretrained("./lora-adapter")
    print("LoRA adapter saved to ./lora-adapter")
