"""Phase 4.2 — Render text generation examples for Countdown GRPO models.

Runs the untrained base model and the trained RL checkpoint on a sample prompt
and saves the output to a JSON file for the UI to display.
"""

import argparse
import json
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

def extract_prompt(sample: dict) -> str:
    """Format the sample into a GRPO system/user prompt."""
    nums = sample["nums"]
    target = sample["target"]
    prompt = (
        f"Using the numbers {nums}, create an equation that equals {target}.\n"
        "You can use basic arithmetic operations (+, -, *, /) and parentheses.\n"
        "Each number can only be used once.\n"
        "Put your final answer inside \u003canswer\u003e...\u003c/answer\u003e tags."
    )
    messages = [
        {"role": "system", "content": "You are a helpful assistant that solves countdown math puzzles."},
        {"role": "user", "content": prompt}
    ]
    return messages

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, help="Path to trained model checkpoint")
    parser.add_argument("--base-model", default="Qwen/Qwen2.5-3B", help="HuggingFace Hub ID of base model")
    parser.add_argument("--output", required=True, help="Path to save output JSON")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[render_countdown] Using device: {device}")

    # Load dataset to get a sample
    print("[render_countdown] Loading dataset...")
    dataset = load_dataset("jianguoz/Countdown-Tasks-3to4", split="train")
    sample = dataset[0] # Pick the first sample
    
    # We will use the chat template of the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    messages = extract_prompt(sample)
    prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt_text, return_tensors="pt").to(device)

    results = {
        "prompt": prompt_text,
        "target": sample["target"],
        "numbers": sample["nums"],
        "base_model_output": "",
        "trained_model_output": ""
    }

    # Generate with base model
    print(f"[render_countdown] Generating with base model {args.base_model}...")
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model, 
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else None
    )
    if device == "cpu":
        base_model = base_model.to(device)

    with torch.no_grad():
        base_outputs = base_model.generate(**inputs, max_new_tokens=256, temperature=0.7)
    
    generated_ids = base_outputs[0][inputs["input_ids"].shape[-1]:]
    results["base_model_output"] = tokenizer.decode(generated_ids, skip_special_tokens=True)
    
    del base_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Generate with trained model
    print(f"[render_countdown] Generating with trained model {args.checkpoint}...")
    try:
        trained_model = AutoModelForCausalLM.from_pretrained(
            args.checkpoint,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            device_map="auto" if device == "cuda" else None
        )
        if device == "cpu":
            trained_model = trained_model.to(device)
            
        with torch.no_grad():
            trained_outputs = trained_model.generate(**inputs, max_new_tokens=256, temperature=0.7)
            
        generated_ids = trained_outputs[0][inputs["input_ids"].shape[-1]:]
        results["trained_model_output"] = tokenizer.decode(generated_ids, skip_special_tokens=True)
        
        del trained_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception as e:
        print(f"[render_countdown] Failed to load or generate with trained model: {e}")
        results["trained_model_output"] = f"Error: {str(e)}"

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[render_countdown] Results saved to {args.output}")

if __name__ == "__main__":
    main()
