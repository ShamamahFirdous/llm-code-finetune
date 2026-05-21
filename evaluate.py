"""
evaluate.py — Compare fine-tuned model vs base Mistral-7B on Python coding tasks
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
import mlflow
import json
from datetime import datetime

MODEL_NAME   = "mistralai/Mistral-7B-v0.1"
ADAPTER_PATH = "./lora-adapter"

# ── Test prompts ─────────────────────────────────────────────────────────────
TEST_PROMPTS = [
    {
        "instruction": "Write a Python function to check if a number is prime.",
        "input": "number = 17",
    },
    {
        "instruction": "Write a Python function to reverse a string.",
        "input": "string = 'hello'",
    },
    {
        "instruction": "Write a Python function to find the factorial of a number.",
        "input": "number = 5",
    },
    {
        "instruction": "Write a Python function to check if a string is a palindrome.",
        "input": "string = 'racecar'",
    },
    {
        "instruction": "Write a Python function to find the maximum element in a list.",
        "input": "lst = [3, 1, 4, 1, 5, 9, 2, 6]",
    },
]

def format_prompt(instruction, input_text):
    return f"""### Instruction:
{instruction}

### Input:
{input_text}

### Response:
"""

def load_model(adapter_path=None):
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
    if adapter_path:
        model = PeftModel.from_pretrained(model, adapter_path)
        print(f"Loaded adapter from {adapter_path}")
    return model, tokenizer

def generate(model, tokenizer, prompt, max_new_tokens=200):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.1,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

def score_response(response):
    """Simple scoring: checks for presence of Python code indicators"""
    score = 0
    if "def " in response:        score += 30
    if "return" in response:      score += 20
    if ":" in response:           score += 10
    if len(response.strip()) > 50: score += 20
    if "```" not in response:     score += 20  # clean output without markdown
    return min(score, 100)

def run_evaluation():
    print("Loading base model...")
    base_model, tokenizer = load_model()

    print("Loading fine-tuned model...")
    ft_model, _ = load_model(adapter_path=ADAPTER_PATH)

    results = []
    base_scores, ft_scores = [], []

    for i, test in enumerate(TEST_PROMPTS):
        prompt = format_prompt(test["instruction"], test["input"])
        print(f"\nTest {i+1}: {test['instruction'][:50]}...")

        base_response = generate(base_model, tokenizer, prompt)
        ft_response   = generate(ft_model, tokenizer, prompt)

        base_score = score_response(base_response)
        ft_score   = score_response(ft_response)

        base_scores.append(base_score)
        ft_scores.append(ft_score)

        results.append({
            "instruction":     test["instruction"],
            "base_response":   base_response,
            "ft_response":     ft_response,
            "base_score":      base_score,
            "ft_score":        ft_score,
            "improvement":     ft_score - base_score,
        })

        print(f"  Base score:        {base_score}/100")
        print(f"  Fine-tuned score:  {ft_score}/100")
        print(f"  Improvement:       {ft_score - base_score:+d}")

    avg_base = sum(base_scores) / len(base_scores)
    avg_ft   = sum(ft_scores)   / len(ft_scores)
    improvement = ((avg_ft - avg_base) / avg_base * 100) if avg_base > 0 else 0

    print(f"\n{'='*50}")
    print(f"Average base score:        {avg_base:.1f}/100")
    print(f"Average fine-tuned score:  {avg_ft:.1f}/100")
    print(f"Overall improvement:       {improvement:.1f}%")

    # Save results
    output = {
        "timestamp":        datetime.now().isoformat(),
        "avg_base_score":   avg_base,
        "avg_ft_score":     avg_ft,
        "improvement_pct":  improvement,
        "results":          results,
    }
    with open("evaluation_results.json", "w") as f:
        json.dump(output, f, indent=2)
    print("\nResults saved to evaluation_results.json")

    # Log to MLflow
    mlflow.set_experiment("llm-code-finetuner")
    with mlflow.start_run(run_name="evaluation"):
        mlflow.log_metric("avg_base_score",  avg_base)
        mlflow.log_metric("avg_ft_score",    avg_ft)
        mlflow.log_metric("improvement_pct", improvement)
        mlflow.log_artifact("evaluation_results.json")

    return output

if __name__ == "__main__":
    run_evaluation()
