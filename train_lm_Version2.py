import argparse
from pathlib import Path
import torch
from tokenizers import Tokenizer
from gpt import GPT, GPTConfig
from tqdm import tqdm

def get_batch(data_ids, block_size, batch_size, device, pad_id):
    ix = torch.randint(0, data_ids.size(0) - block_size - 1, (batch_size,))
    x = torch.stack([data_ids[i:i+block_size] for i in ix])
    y = torch.stack([data_ids[i+1:i+1+block_size] for i in ix])
    return x.to(device), y.to(device)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifacts_dir", type=str, default="artifacts")
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--out_dir", type=str, default="checkpoints/pretrain")
    parser.add_argument("--block_size", type=int, default=256)
    parser.add_argument("--n_layer", type=int, default=6)
    parser.add_argument("--n_head", type=int, default=6)
    parser.add_argument("--n_embd", type=int, default=384)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--max_steps", type=int, default=5000)
    parser.add_argument("--eval_interval", type=int, default=500)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    tokenizer = Tokenizer.from_file(str(Path(args.artifacts_dir) / "tokenizer.json"))
    vocab_size = tokenizer.get_vocab_size()
    pad_id = tokenizer.token_to_id("[PAD]")
    eos_id = tokenizer.token_to_id("[EOS]")

    train_ids = torch.load(Path(args.data_dir) / "train_ids.pt", map_location="cpu").long()
    val_ids = torch.load(Path(args.data_dir) / "val_ids.pt", map_location="cpu").long()

    cfg = GPTConfig(
        vocab_size=vocab_size,
        block_size=args.block_size,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
        dropout=args.dropout,
        pad_id=pad_id,
    )
    model = GPT(cfg).to(args.device)
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.1)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    def evaluate():
        model.eval()
        with torch.no_grad():
            losses = []
            for _ in range(50):
                xb, yb = get_batch(val_ids, args.block_size, args.batch_size, args.device, pad_id)
                _, loss = model(xb, yb)
                losses.append(loss.item())
            return sum(losses)/len(losses)

    best_val = float("inf")
    pbar = tqdm(range(1, args.max_steps + 1), desc="pretrain")
    for step in pbar:
        model.train()
        xb, yb = get_batch(train_ids, args.block_size, args.batch_size, args.device, pad_id)
        _, loss = model(xb, yb)
        optim.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optim.step()

        if step % args.eval_interval == 0 or step == 1:
            val_loss = evaluate()
            pbar.set_postfix(loss=loss.item(), val=val_loss)
            if val_loss < best_val:
                best_val = val_loss
                ckpt = {
                    "model": model.state_dict(),
                    "config": cfg.__dict__,
                    "step": step,
                }
                torch.save(ckpt, out_dir / "best.pt")
    # save final
    ckpt = {
        "model": model.state_dict(),
        "config": cfg.__dict__,
        "step": args.max_steps,
    }
    torch.save(ckpt, out_dir / "last.pt")
    print(f"Saved checkpoints to {out_dir}")

if __name__ == "__main__":
    main()