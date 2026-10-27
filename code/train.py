import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments


IGNORE_INDEX = -100
SYSTEM_PROMPT = "You are a medical reasoning assistant. Answer the multiple-choice medical question with concise evidence-based reasoning and a final answer."


def read_jsonl(path):
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def format_prompt(tokenizer, prompt):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"{prompt.strip()}\n\nAnswer:"},
    ]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )


def tokenize_pair(tokenizer, prompt, response, max_length):
    prompt_ids = tokenizer(
        format_prompt(tokenizer, prompt),
        add_special_tokens=False,
    ).input_ids
    response_ids = tokenizer(
        response.strip(),
        add_special_tokens=False,
    ).input_ids + [tokenizer.eos_token_id]
    input_ids = prompt_ids + response_ids
    labels = [IGNORE_INDEX] * len(prompt_ids) + response_ids
    if len(input_ids) > max_length:
        input_ids = input_ids[-max_length:]
        labels = labels[-max_length:]
    return {
        "input_ids": input_ids,
        "attention_mask": [1] * len(input_ids),
        "labels": labels,
    }


class PreferenceDataset(Dataset):
    def __init__(self, path, tokenizer, max_length):
        self.rows = read_jsonl(path)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        chosen = tokenize_pair(
            self.tokenizer,
            row["prompt"],
            row["chosen"],
            self.max_length,
        )
        rejected = tokenize_pair(
            self.tokenizer,
            row["prompt"],
            row["rejected"],
            self.max_length,
        )
        return {
            "chosen_input_ids": chosen["input_ids"],
            "chosen_attention_mask": chosen["attention_mask"],
            "chosen_labels": chosen["labels"],
            "rejected_input_ids": rejected["input_ids"],
            "rejected_attention_mask": rejected["attention_mask"],
            "rejected_labels": rejected["labels"],
            "preference_weight": float(row["preference_weight"]),
        }


@dataclass
class PreferenceCollator:
    tokenizer: object

    def pad(self, features, key, value):
        length = max(len(item[key]) for item in features)
        return torch.tensor(
            [item[key] + [value] * (length - len(item[key])) for item in features],
            dtype=torch.long,
        )

    def __call__(self, features):
        return {
            "chosen_input_ids": self.pad(
                features,
                "chosen_input_ids",
                self.tokenizer.pad_token_id,
            ),
            "chosen_attention_mask": self.pad(
                features,
                "chosen_attention_mask",
                0,
            ),
            "chosen_labels": self.pad(
                features,
                "chosen_labels",
                IGNORE_INDEX,
            ),
            "rejected_input_ids": self.pad(
                features,
                "rejected_input_ids",
                self.tokenizer.pad_token_id,
            ),
            "rejected_attention_mask": self.pad(
                features,
                "rejected_attention_mask",
                0,
            ),
            "rejected_labels": self.pad(
                features,
                "rejected_labels",
                IGNORE_INDEX,
            ),
            "preference_weight": torch.tensor(
                [item["preference_weight"] for item in features],
                dtype=torch.float,
            ),
        }


def sequence_logps(logits, labels):
    logits = logits[:, :-1]
    labels = labels[:, 1:]
    mask = labels.ne(IGNORE_INDEX)
    safe_labels = labels.masked_fill(~mask, 0)
    token_logps = -F.cross_entropy(
        logits.transpose(1, 2),
        safe_labels,
        reduction="none",
    )
    return (token_logps * mask).sum(-1) / mask.sum(-1).clamp_min(1)


def log_odds(logps):
    logps = logps.float().clamp(max=-1e-7)
    return logps - torch.log1p(-torch.exp(logps))


def token_loss(logits, labels):
    logits = logits[:, :-1].contiguous()
    labels = labels[:, 1:].contiguous()
    return F.cross_entropy(
        logits.view(-1, logits.size(-1)),
        labels.view(-1),
        ignore_index=IGNORE_INDEX,
    )


class MCPOTrainer(Trainer):
    def __init__(self, *args, lambda_pref=0.2, **kwargs):
        super().__init__(*args, **kwargs)
        self.lambda_pref = lambda_pref

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        chosen = model(
            input_ids=inputs["chosen_input_ids"],
            attention_mask=inputs["chosen_attention_mask"],
        )
        sft_loss = token_loss(chosen.logits, inputs["chosen_labels"])
        chosen_logps = sequence_logps(chosen.logits, inputs["chosen_labels"])
        rejected = model(
            input_ids=inputs["rejected_input_ids"],
            attention_mask=inputs["rejected_attention_mask"],
        )
        rejected_logps = sequence_logps(rejected.logits, inputs["rejected_labels"])
        preference = -F.logsigmoid(log_odds(chosen_logps) - log_odds(rejected_logps))
        weights = inputs["preference_weight"].to(preference.device)
        preference_loss = (weights * preference).mean()
        loss = sft_loss + self.lambda_pref * preference_loss
        if return_outputs:
            return loss, {
                "sft_loss": sft_loss.detach(),
                "preference_loss": preference_loss.detach(),
            }
        return loss


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--train-file", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--deepspeed")
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--epochs", type=float, default=3)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--lambda-pref", type=float, default=0.2)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        trust_remote_code=True,
        use_fast=False,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    model.gradient_checkpointing_enable()
    model.config.use_cache = False
    dataset = PreferenceDataset(args.train_file, tokenizer, args.max_length)
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        learning_rate=args.learning_rate,
        num_train_epochs=args.epochs,
        warmup_ratio=args.warmup_ratio,
        bf16=True,
        deepspeed=args.deepspeed,
        gradient_checkpointing=True,
        remove_unused_columns=False,
        report_to="none",
        save_strategy="epoch",
        logging_steps=5,
    )
    trainer = MCPOTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=PreferenceCollator(tokenizer),
        lambda_pref=args.lambda_pref,
    )
    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()
