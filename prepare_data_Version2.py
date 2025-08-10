import os
import argparse
import requests
from pathlib import Path
from tokenizers import Tokenizer, models, trainers, pre_tokenizers
import torch
import numpy as np
from tqdm import tqdm
import random

TINY_SHAKESPEARE_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"

SPECIAL_TOKENS = ["[PAD]", "[BOS]", "[EOS]", "[UNK]", "[USER]", "[ASSISTANT]", "[SYSTEM]"]

def download_tiny_shakespeare(dst_path: Path):
    if dst_path.exists():
        print(f"Found {dst_path}")
        return
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    print("Downloading tiny Shakespeare...")
    r = requests.get(TINY_SHAKESPEARE_URL, timeout=30)
    r.raise_for_status()
    dst_path.write_text(r.text, encoding="utf-8")
    print(f"Saved to {dst_path}")

def train_tokenizer(corpus_files, vocab_size, out_path: Path):
    print("Training tokenizer...")
    tokenizer = Tokenizer(models.BPE(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=2,
        special_tokens=SPECIAL_TOKENS,
        show_progress=True,
    )
    tokenizer.train(corpus_files=corpus_files, trainer=trainer)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tokenizer.save(str(out_path))
    print(f"Saved tokenizer to {out_path}")
    return out_path

def encode_file(tokenizer_path: Path, text_path: Path):
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    text = Path(text_path).read_text(encoding="utf-8")
    ids = tokenizer.encode(text).ids
    return np.array(ids, dtype=np.int32)

def split_and_save_ids(ids: np.ndarray, train_ratio: float, out_dir: Path):
    n = len(ids)
    n_train = int(n * train_ratio)
    train_ids = ids[:n_train]
    val_ids = ids[n_train:]
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(torch.from_numpy(train_ids.astype(np.int32)), out_dir / "train_ids.pt")
    torch.save(torch.from_numpy(val_ids.astype(np.int32)), out_dir / "val_ids.pt")
    print(f"Saved train/val ids to {out_dir} | train: {len(train_ids)} val: {len(val_ids)}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--artifacts_dir", type=str, default="artifacts")
    parser.add_argument("--vocab_size", type=int, default=32000)
    parser.add_argument("--train_ratio", type=float, default=0.9)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    artifacts_dir = Path(args.artifacts_dir)
    raw_txt = data_dir / "raw" / "tinyshakespeare.txt"
    download_tiny_shakespeare(raw_txt)

    tokenizer_path = artifacts_dir / "tokenizer.json"
    train_tokenizer([str(raw_txt)], args.vocab_size, tokenizer_path)

    ids = encode_file(tokenizer_path, raw_txt)
    split_and_save_ids(ids, args.train_ratio, data_dir)

if __name__ == "__main__":
    main()