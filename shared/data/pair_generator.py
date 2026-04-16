"""
Contrastive pair construction for arithmetic faithfulness experiments.

Generates paired clean/corrupted prompts across multiple arithmetic
task types (2-digit addition, 3-digit addition, subtraction, word
problems) with metadata for carry flags, corruption severity, etc.
"""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Tuple

import numpy as np


@dataclass
class ContrastivePair:
    """A single contrastive pair for activation patching."""

    pair_id: str
    clean_prompt: str
    corrupted_prompt: str
    correct_token: str
    incorrect_token: str
    pair_type: str
    metadata: Dict[str, Any] = field(default_factory=dict)


# ── Arithmetic pair factories ──────────────────────────────────────────


def _make_2digit_addition_pair(
    pair_idx: int,
    rng: random.Random,
    carry: bool = False,
) -> ContrastivePair:
    """Generate a 2-digit addition pair (faithful vs corrupted CoT)."""
    while True:
        a = rng.randint(10, 99)
        b = rng.randint(10, 99)
        units_sum = (a % 10) + (b % 10)
        has_carry = units_sum >= 10
        if has_carry == carry:
            break

    correct = a + b
    a_u, a_t = a % 10, a // 10
    b_u, b_t = b % 10, b // 10
    units = a_u + b_u
    tens = a_t + b_t

    clean = (
        f"Q: What is {a}+{b}?\n"
        f"Steps: units={a_u}+{b_u}={units}, tens={a_t}+{b_t}={tens}.\n"
        f"A:"
    )

    corruption = rng.choice(["units", "tens"])
    if corruption == "units":
        wrong_u = units + rng.choice([3, 5, 7, -3, -5])
        wrong_t = tens
    else:
        wrong_u = units
        wrong_t = tens + rng.choice([2, 3, 4, -2, -3])

    wrong_answer = wrong_t * 10 + wrong_u
    if wrong_answer == correct:
        wrong_answer += 10
        wrong_t += 1

    corrupted = (
        f"Q: What is {a}+{b}?\n"
        f"Steps: units={a_u}+{b_u}={wrong_u}, tens={a_t}+{b_t}={wrong_t}.\n"
        f"A:"
    )

    pid = hashlib.md5(f"2dadd_{pair_idx}_{a}_{b}".encode()).hexdigest()[:12]

    return ContrastivePair(
        pair_id=pid,
        clean_prompt=clean,
        corrupted_prompt=corrupted,
        correct_token=f" {correct}",
        incorrect_token=f" {wrong_answer}",
        pair_type="2digit_addition",
        metadata={
            "a": a,
            "b": b,
            "correct": correct,
            "carry_required": carry,
            "corruption_type": f"{corruption}_error",
            "corruption_severity": abs(wrong_answer - correct) / max(correct, 1),
        },
    )


def _make_3digit_addition_pair(pair_idx: int, rng: random.Random) -> ContrastivePair:
    a = rng.randint(100, 999)
    b = rng.randint(100, 999)
    correct = a + b

    clean = f"Q: What is {a}+{b}?\nA: Let me add step by step. {a}+{b}={correct}\nA:"
    wrong = correct + rng.choice([10, 20, -10, -20, 100, -100])
    corrupted = f"Q: What is {a}+{b}?\nA: Let me add step by step. {a}+{b}={wrong}\nA:"

    pid = hashlib.md5(f"3dadd_{pair_idx}_{a}_{b}".encode()).hexdigest()[:12]
    return ContrastivePair(
        pair_id=pid,
        clean_prompt=clean,
        corrupted_prompt=corrupted,
        correct_token=f" {correct}",
        incorrect_token=f" {wrong}",
        pair_type="3digit_addition",
        metadata={"a": a, "b": b, "correct": correct, "carry_required": True},
    )


def _make_subtraction_pair(pair_idx: int, rng: random.Random) -> ContrastivePair:
    a = rng.randint(30, 99)
    b = rng.randint(10, a - 1)
    correct = a - b

    clean = f"Q: What is {a}-{b}?\nSteps: {a}-{b}={correct}.\nA:"
    wrong = correct + rng.choice([3, 5, -3, -5, 10, -10])
    corrupted = f"Q: What is {a}-{b}?\nSteps: {a}-{b}={wrong}.\nA:"

    pid = hashlib.md5(f"sub_{pair_idx}_{a}_{b}".encode()).hexdigest()[:12]
    return ContrastivePair(
        pair_id=pid,
        clean_prompt=clean,
        corrupted_prompt=corrupted,
        correct_token=f" {correct}",
        incorrect_token=f" {wrong}",
        pair_type="2digit_subtraction",
        metadata={"a": a, "b": b, "correct": correct},
    )


def _make_word_problem_pair(pair_idx: int, rng: random.Random) -> ContrastivePair:
    a = rng.randint(15, 50)
    b = rng.randint(5, 30)
    correct = a + b

    clean = (
        f"Q: Sarah has {a} apples and buys {b} more. How many does she have?\n"
        f"Steps: {a}+{b}={correct}.\nA:"
    )
    wrong = correct + rng.choice([3, 5, -3, -5])
    corrupted = (
        f"Q: Sarah has {a} apples and buys {b} more. How many does she have?\n"
        f"Steps: {a}+{b}={wrong}.\nA:"
    )

    pid = hashlib.md5(f"wp_{pair_idx}_{a}_{b}".encode()).hexdigest()[:12]
    return ContrastivePair(
        pair_id=pid,
        clean_prompt=clean,
        corrupted_prompt=corrupted,
        correct_token=f" {correct}",
        incorrect_token=f" {wrong}",
        pair_type="word_problem",
        metadata={"a": a, "b": b, "correct": correct},
    )


# ── Public API ─────────────────────────────────────────────────────────

TASK_FACTORIES = {
    "2digit_add_no_carry": lambda idx, rng: _make_2digit_addition_pair(idx, rng, carry=False),
    "2digit_add_carry": lambda idx, rng: _make_2digit_addition_pair(idx, rng, carry=True),
    "3digit_add": _make_3digit_addition_pair,
    "2digit_sub": _make_subtraction_pair,
    "word_problem": _make_word_problem_pair,
}


def generate_arithmetic_dataset(
    n_pairs: int = 2000,
    task_types: List[str] | None = None,
    seed: int = 42,
) -> List[ContrastivePair]:
    """Generate a balanced dataset of contrastive arithmetic pairs.

    Parameters
    ----------
    n_pairs
        Total number of pairs to generate.  Distributed evenly across
        *task_types*.
    task_types
        Subset of keys from ``TASK_FACTORIES``.  Defaults to all types.
    seed
        Random seed for reproducibility.
    """
    if task_types is None:
        task_types = list(TASK_FACTORIES.keys())

    rng = random.Random(seed)
    per_type = n_pairs // len(task_types)
    remainder = n_pairs - per_type * len(task_types)

    pairs: List[ContrastivePair] = []
    for i, tt in enumerate(task_types):
        factory = TASK_FACTORIES[tt]
        count = per_type + (1 if i < remainder else 0)
        for j in range(count):
            pairs.append(factory(len(pairs), rng))

    rng.shuffle(pairs)
    return pairs
