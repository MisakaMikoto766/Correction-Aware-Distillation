import argparse
import json
import re
import time
from collections import defaultdict
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


DATASETS = ["MedMCQA", "MedQA", "PubMedQA"]
SYSTEM_PROMPT = "You are a medical reasoning assistant. Answer the multiple-choice medical question with concise evidence-based reasoning and a final answer."


def read_jsonl(path):
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def format_prompt(tokenizer, row):
    options = "\n".join(
        f"{key}. {value}" for key, value in sorted(row["options"].items())
    )
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": f"{row['question']}\n\n{options}\n\nAnswer:",
        },
    ]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )


def extract_answer(text, valid):
    patterns = [
        r"(?:final|correct|best)\s+answer\s*(?:is|:)\s*(?:option\s*)?\(?([A-Z])\)?",
        r"\boption\s*\(?([A-Z])\)?\s+(?:is\s+)?(?:the\s+)?correct\b",
        r"\b([A-Z])\s*[\.:：]",
    ]
    for pattern in patterns:
        matches = [
            match.group(1).upper()
            for match in re.finditer(pattern, text, flags=re.IGNORECASE)
            if match.group(1).upper() in valid
        ]
        if matches:
            return matches[-1]
    first = text.strip()[:1].upper()
    return first if first in valid else ""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--test-file", required=True)
    parser.add_argument("--output-file", required=True)
    parser.add_argument("--summary-file", required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-input-length", type=int, default=2048)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    args = parser.parse_args()

    rows = read_jsonl(args.test_file)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        trust_remote_code=True,
        use_fast=False,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    ).eval()

    output = []
    generated_tokens = 0
    started = time.perf_counter()
    for start in range(0, len(rows), args.batch_size):
        batch = rows[start : start + args.batch_size]
        prompts = [format_prompt(tokenizer, row) for row in batch]
        inputs = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=args.max_input_length,
        ).to(model.device)
        with torch.inference_mode():
            generated = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        new_tokens = generated[:, inputs["input_ids"].shape[1] :]
        texts = tokenizer.batch_decode(new_tokens, skip_special_tokens=True)
        generated_tokens += sum(
            len(tokenizer.encode(text, add_special_tokens=False)) for text in texts
        )
        for row, text in zip(batch, texts):
            prediction = extract_answer(text, set(row["options"]))
            output.append(
                {
                    "sample_id": row["sample_id"],
                    "dataset": row["dataset"],
                    "prediction": prediction,
                    "answer": row["answer_idx"],
                    "correct": prediction == row["answer_idx"],
                    "generated_text": text.strip(),
                }
            )
    elapsed = time.perf_counter() - started

    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for row in output:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    counts = defaultdict(lambda: [0, 0])
    for row in output:
        counts[row["dataset"]][0] += int(row["correct"])
        counts[row["dataset"]][1] += 1
    accuracies = {
        dataset: counts[dataset][0] / counts[dataset][1]
        for dataset in DATASETS
    }
    total_correct = sum(counts[dataset][0] for dataset in DATASETS)
    summary = {
        "accuracy": accuracies,
        "overall_accuracy": total_correct / len(output),
        "macro_accuracy": sum(accuracies.values()) / len(DATASETS),
        "average_calls": 1.0,
        "average_generated_tokens": generated_tokens / len(output),
        "average_latency_seconds": elapsed / len(output),
    }
    summary_path = Path(args.summary_file)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
