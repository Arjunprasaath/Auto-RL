"""
Countdown puzzle environment for AutoRL GRPO training.

The task: given a list of numbers, use +, -, *, / (each number exactly once)
to reach a target number.

Functions:
    generate_countdown_prompt(numbers, target) -> str
    format_reward_fn(completions, prompts, **kwargs) -> list[float]
    accuracy_reward_fn(completions, prompts, target, numbers, **kwargs) -> list[float]
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

# TinyZero system prompt: tells the model to use <think>/<answer> tags.
# The format reward reinforces this structure, making before/after visible.
SYSTEM_PROMPT = (
    "A conversation between User and Assistant. The user asks a question, and the "
    "Assistant solves it. The assistant first thinks about the reasoning process in "
    "the mind and then provides the user with the answer. The reasoning process and "
    "answer are enclosed within <think> </think> and <answer> </answer> tags, "
    "respectively, i.e., <think> reasoning process here </think> "
    "<answer> answer here </answer>."
)


def generate_countdown_prompt(numbers: list[int], target: int) -> str:
    """Format a Countdown puzzle as the user message."""
    return (
        f"Using the numbers {numbers} with operations +, -, *, / "
        f"(each number used exactly once), reach the target: {target}. "
        f"Think step by step, then give your final expression in <answer> tags."
    )


def _extract_text(completion) -> str:
    """Normalize a completion to a plain string regardless of format."""
    if isinstance(completion, list):
        return completion[-1].get("content", "") if completion else ""
    if isinstance(completion, dict):
        return completion.get("content", "")
    return str(completion)


def format_reward_fn(completions, prompts, **kwargs) -> list[float]:
    """
    Reward 0.1 when the completion contains both <think>...</think> and
    <answer>...</answer> tags. Fires early in training to teach structure
    before accuracy improves.
    """
    rewards = []
    for completion in completions:
        text = _extract_text(completion)
        has_think = bool(re.search(r"<think>.*?</think>", text, re.DOTALL))
        has_answer = bool(re.search(r"<answer>.*?</answer>", text, re.DOTALL))
        rewards.append(0.1 if (has_think and has_answer) else 0.0)
    return rewards


def accuracy_reward_fn(
    completions, prompts, target: list[int], numbers: list[list[int]], **kwargs
) -> list[float]:
    """
    Binary reward: 1.0 if the expression inside <answer> tags evaluates to
    target, 0.0 otherwise. Falls back to searching the full text if no tags
    are present.
    """
    rewards = []
    for completion, t, nums in zip(completions, target, numbers):
        text = _extract_text(completion)
        m = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL)
        answer_text = m.group(1).strip() if m else text
        score = evaluate_solution(answer_text, t, nums)
        rewards.append(1.0 if score == 1.0 else 0.0)
    return rewards


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
