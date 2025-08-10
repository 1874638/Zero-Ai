import os
import argparse
import json
from typing import Dict, List

import torch
from datasets import load_dataset, Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig
from trl import SFTTrainer

os.environ["TOKENIZERS_PARALLELISM"] = "false"

PROMPT_TMPL_WITH_INPUT = """### Instruction:
{instruction}

### Input:
{input}

### Response:
"""
PROMPT_TMPL_NO_INPUT = """### Instruction:
{instruction}

### Response:
"""

def load_jsonl_dataset(path: str) -> Dataset:
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                obj = json.loads(line)
                # Expect fields: instruction, input (optional), output
                records.append(obj)
    return Dataset.from_list(records)

def build_formatting_func():
    def fmt(samples: List[Dict]) -> List[str]:
        texts = []
        for ex in samples:
            instruction = ex.get("instruction", "").strip()
            input_text = ex.get("input", "")
            output = ex.get("output", "").strip()
            if input_text and len(input_text.strip()) > 0:
                prompt = PROMPT_TMPL_WITH_INPUT.format(instruction=instruction, input=input_text.strip())
            else:
                prompt = PROMPT_TMPL_NO_INPUT.format(instruction=instruction)
            # Trainer will mask labels before "### Response:\n" using response_template
            texts.append(prompt + output)
        return texts
    return fmt

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", type=str, default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    parser.add_argument("--dataset_path", type=str, default="sample_dataset.jsonl")
    parser.add_argument("--output_dir", type=str, default="./outputs/qlora-tinyllama")
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--per_device_batch_size", type=int, default=2)
    parser.add_argument("--grad_accum_steps", type=int, default=8)
    parser.add_argument("--max_seq_len", type=int, default=2048)
    parser.add_argument("--use_bf16", action="store_true")
    args = parser.parse_args()

    compute_dtype = torch.bfloat16 if args.use_bf16 and torch.cuda.is_available() else torch.float16

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=compute_dtype,
    )

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    model.config.use_cache = False  # with gradient checkpointing

    # Data
    if args.dataset_path.endswith(".jsonl"):
        dataset = load_jsonl_dataset(args.dataset_path)
    else:
        dataset = load_dataset(args.dataset_path)["train"]  # supports HF hub datasets too

    # LoRA config (you can tune r/alpha/target_modules as needed)
    lora_cfg = LoraConfig(
        r=64,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )

    train_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_batch_size,
        gradient_accumulation_steps=args.grad_accum_steps,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        logging_steps=10,
        save_steps=500,
        save_total_limit=2,
        bf16=(compute_dtype == torch.bfloat16),
        fp16=(compute_dtype == torch.float16),
        optim="paged_adamw_8bit",
        gradient_checkpointing=True,
        report_to="none",
    )

    formatter = build_formatting_func()

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        peft_config=lora_cfg,
        args=train_args,
        max_seq_length=args.max_seq_len,
        packing=False,
        formatting_func=formatter,
        dataset_num_proc=4,
        response_template="### Response:\n",
    )

    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    print(f"Training complete. Adapter saved to {args.output_dir}")

if __name__ == "__main__":
    main()