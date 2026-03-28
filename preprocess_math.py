import json
import re
import datasets


CONFIGS = [
    "algebra",
    "counting_and_probability",
    "geometry",
    "intermediate_algebra",
    "number_theory",
    "prealgebra",
    "precalculus",
]

VALID_RATIO = 0.05


def extract_boxed_answer(solution: str) -> str:
    """Extract the final answer from \\boxed{...}, handling nested braces."""
    idx = solution.rfind("\\boxed{")
    if idx == -1:
        return ""
    start = idx + len("\\boxed{")
    depth = 1
    i = start
    while i < len(solution) and depth > 0:
        if solution[i] == "{":
            depth += 1
        elif solution[i] == "}":
            depth -= 1
        i += 1
    return solution[start : i - 1]


def split_solution_into_steps(solution: str) -> list[str]:
    """Split a MATH solution into reasoning steps (sentence-level)."""
    steps = re.split(r"(?<=[.!?])\s+", solution.strip())
    steps = [s.strip() for s in steps if s.strip()]
    return steps


def convert_example(example: dict) -> dict:
    """Convert a single MATH example to the GSM-like format."""
    return {
        "question": example["problem"],
        "steps": split_solution_into_steps(example["solution"]),
        "answer": extract_boxed_answer(example["solution"]),
    }


def main():
    all_train = []
    all_test = []

    for cfg in CONFIGS:
        print(f"Loading {cfg}...")
        ds = datasets.load_dataset("EleutherAI/hendrycks_math", cfg)
        all_train.extend(convert_example(ex) for ex in ds["train"])
        all_test.extend(convert_example(ex) for ex in ds["test"])

    print(f"Total train (before split): {len(all_train)}")
    print(f"Total test: {len(all_test)}")

    n_valid = int(len(all_train) * VALID_RATIO)
    valid_data = all_train[:n_valid]
    train_data = all_train[n_valid:]

    print(f"Train: {len(train_data)}, Valid: {len(valid_data)}, Test: {len(all_test)}")

    json.dump(train_data, open("data/math_train.json", "w"))
    json.dump(valid_data, open("data/math_valid.json", "w"))
    json.dump(all_test, open("data/math_test.json", "w"))
    print("Saved data/math_train.json, data/math_valid.json, data/math_test.json")


if __name__ == "__main__":
    main()
