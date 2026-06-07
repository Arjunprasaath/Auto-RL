"""
Countdown puzzle environment for AutoRL GRPO training.

The task: given a list of numbers, use +, -, *, / (each number exactly once)
to reach a target number.

Functions:
    generate_countdown_prompt(numbers, target) -> str
    evaluate_solution(model_output, target, numbers) -> float
    load_countdown_dataset() -> Dataset
"""

import re
import operator
from datasets import load_dataset

OPS = {
    "+": operator.add,
    "-": operator.sub,
    "*": operator.mul,
    "/": operator.truediv,
}

DATASET_NAME = "Jiayi-Pan/Countdown-Tasks-3to4"


def generate_countdown_prompt(numbers: list[int], target: int) -> str:
    """Format a Countdown puzzle as a prompt for the LLM."""
    return (
        f"Using the numbers {numbers} with operations +, -, *, / "
        f"(each number used exactly once), reach the target: {target}\n"
        f"Show your work step by step, then write the final expression."
    )


def evaluate_solution(model_output: str, target: int, numbers: list[int]) -> float:
    """
    Score a model's response to a Countdown puzzle.

    Returns:
        1.0  — expression found that evaluates exactly to target
        0.5  — valid arithmetic expression found but wrong result
        0.0  — no valid expression found

    Uses regex to extract digit/operator patterns before eval() so only
    arithmetic expressions are evaluated. Safe for controlled RunPod environment.
    """
    expressions = re.findall(r"[\d\s\+\-\*\/\(\)\.]+", model_output)

    for expr in reversed(expressions):
        expr = expr.strip()
        if not expr or not any(c.isdigit() for c in expr) or "**" in expr:
            continue
        try:
            result = eval(expr)  # noqa: S307 — controlled environment
            if isinstance(result, (int, float)) and abs(result - target) < 1e-6:
                return 1.0
            elif isinstance(result, (int, float)):
                return 0.5
        except Exception:
            continue

    return 0.0


def load_countdown_dataset(split: str = "train", test_ratio: float = 0.05, seed: int = 42):
    """
    Load the Countdown dataset from HuggingFace.
    Dataset: Jiayi-Pan/Countdown-Tasks-3to4
    Each row has 'nums' (list of ints) and 'target' (int).

    The dataset has no built-in train/test split, so we create one:
    - split="train" → 95% of data (shuffled, seeded)
    - split="test"  → 5% of data (held-out)
    """
    full = load_dataset(DATASET_NAME, split="train")
    splits = full.train_test_split(test_size=test_ratio, seed=seed)
    return splits[split]
