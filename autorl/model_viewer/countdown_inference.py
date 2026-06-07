"""
Countdown inference — load a GRPO LoRA checkpoint and solve test puzzles.

Outputs countdown_solve.json for the UI ModelViewer component, which
renders the model's chain-of-thought line by line with animated cards.

Run from autorl/ package root:
    python model_viewer/countdown_inference.py \
        --checkpoint results/agent_3/checkpoint \
        --n 5 --output results/countdown_solve.json
"""

import argparse
import json
import os
import sys

_PKG_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PKG_ROOT not in sys.path:
    sys.path.insert(0, _PKG_ROOT)

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from environments.countdown_env import (
    evaluate_solution,
    generate_countdown_prompt,
    load_countdown_dataset,
)

BASE_MODEL = "Qwen/Qwen2.5-3B"


def run_inference(
    checkpoint_path: str,
    n: int = 5,
    output_path: str = "results/countdown_solve.json",
    seed: int = 42,
) -> list[dict]:
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, torch_dtype="auto")
    model = PeftModel.from_pretrained(model, checkpoint_path)
    model.eval()

    test_data = list(load_countdown_dataset(split="test", seed=seed))[:n]

    results = []
    for row in test_data:
        prompt = generate_countdown_prompt(row["nums"], row["target"])
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            output = model.generate(**inputs, max_new_tokens=300, do_sample=False)
        full_text = tokenizer.decode(output[0], skip_special_tokens=True)
        response = full_text[len(prompt):]

        score = evaluate_solution(response, row["target"], row["nums"])
        results.append({
            "numbers": row["nums"],
            "target": row["target"],
            "prompt": prompt,
            "model_response": response,
            "success": score == 1.0,
        })
        status = "ok" if score == 1.0 else "fail"
        print(f"[{row['nums']} -> {row['target']}] {status}")

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved {len(results)} results to {output_path}")
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--n", type=int, default=5)
    parser.add_argument("--output", default="results/countdown_solve.json")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    run_inference(args.checkpoint, args.n, args.output, args.seed)


if __name__ == "__main__":
    main()
