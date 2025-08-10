import argparse
import json
from pathlib import Path
from typing import List, Dict

import torch
from torch.utils.data import Dataset, DataLoader
from tokenizers import Tokenizer
from gpt import GPT, GPTConfig
from tqdm import tqdm

SYSTEM_PROMPT = "You are a helpful assistant."
USER_TAG = "[USER]"
ASSIST_TAG = "[ASSISTANT]"
SYS_TAG = "[SYSTEM]"
EOS_TAG = "[EOS]"

def format_example(ex: Dict) -> str:
    instruction = ex.get("instruction", "").strip()
    input_text = ex.get("input", "").strip()
    response = ex.get("output", "").strip()
    if input_text:
        prompt = f"{SYS_TAG} {SYSTEM_PROMPT}\n{USER_TAG} {instruction}\n{input_text}\n{ASSIST_TAG} "
    else:
        prompt = f"{SYS_TAG} {SYSTEM_PROMPT}\n{USER_TAG} {instruction}\n{ASSIST_TAG} "
    return prompt, response

class SFTJsonlDataset(Dataset):
    def __init__(self, tokenizer: Tokenizer, jsonl_path: Path, block_size: int):
        self.tokenizer = tokenizer
        self.block_size = block_size
        self.pad_id = tokenizer.token_to_id("[PAD]")
        self.eos_id = tokenizer.token_to_id("[EOS]")
        self.items = []
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    ex = json.loads(line)
                    prompt, response = format_example(ex)
                    full = prompt + response + f" {EOS_TAG}"
                    enc = tokenizer.encode(full)
                    ids = enc.ids[:block_size]
                    # labels: mask everything before assistant start
                    assist_pos = len(tokenizer.encode(prompt).ids)
                    labels = [-100] * assist_pos + ids[assist_pos:]
                    labels = labels[:len(ids)]
                    self.items.append((ids, labels))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        ids, labels = self.items[idx]
        x = torch.tensor(ids, dtype=torch.long)
        y = torch.tensor(labels, dtype=torch.long)
        # pad to block_size
        if len(ids) < self.block_size:
            pad_len = self.block_size - len(ids)
            x = torch.nn.functional.pad(x, (0, pad_len), value=self.pad_id)
            y = torch.nn.functional.pad(y, (0, pad_len), value=-100)
        return x, y

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifacts_dir", type=str, default="artifacts")
    parser.add_argument("--pretrain_ckpt", type=str, default="checkpoints/pretrain/best.pt")
    parser.add_argument("--sft_data", type=str, default="sft_data.jsonl")
    parser.add_argument("--out_dir", type=str, default="checkpoints/sft")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    tokenizer = Tokenizer.from_file(str(Path(args.artifacts_dir) / "tokenizer.json"))
    vocab_size = tokenizer.get_vocab_size()
    pad_id = tokenizer.token_to_id("[PAD]")

    ckpt = torch.load(args.pretrain_ckpt, map_location="cpu")
    cfg_dict = ckpt["config"]
    cfg = GPTConfig(**cfg_dict)
    model = GPT(cfg)
    model.load_state_dict(ckpt["model"], strict=True)
    model.to(args.device)

    dataset = SFTJsonlDataset(tokenizer, Path(args.sft_data), cfg.block_size)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=False)

    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.1)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    global_step = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        pbar = tqdm(loader, desc=f"sft epoch {epoch}")
        for xb, yb in pbar:
            xb = xb.to(args.device)
            yb = yb.to(args.device)
            _, loss = model(xb, yb)
            optim.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
            global_step += 1
            pbar.set_postfix(loss=loss.item())

        # save each epoch
        torch.save(
            {"model": model.state_dict(), "config": cfg.__dict__, "epoch": epoch, "step": global_step},
            out_dir / f"epoch{epoch}.pt",
        )
    print(f"Saved SFT checkpoints to {out_dir}")

if __name__ == "__main__":
    main()