import argparse
from pathlib import Path
import torch
from tokenizers import Tokenizer
from gpt import GPT, GPTConfig

SYS_TAG = "[SYSTEM]"
USER_TAG = "[USER]"
ASSIST_TAG = "[ASSISTANT]"
BOS_TAG = "[BOS]"
EOS_TAG = "[EOS]"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifacts_dir", type=str, default="artifacts")
    parser.add_argument("--ckpt", type=str, default="checkpoints/sft/epoch3.pt")
    parser.add_argument("--max_new_tokens", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--system", type=str, default="You are a helpful assistant.")
    args = parser.parse_args()

    tokenizer = Tokenizer.from_file(str(Path(args.artifacts_dir) / "tokenizer.json"))
    pad_id = tokenizer.token_to_id("[PAD]")
    eos_id = tokenizer.token_to_id("[EOS]")

    ckpt = torch.load(args.ckpt, map_location="cpu")
    cfg = GPTConfig(**ckpt["config"])
    model = GPT(cfg)
    model.load_state_dict(ckpt["model"])
    model.to(args.device)
    model.eval()

    history = f"{SYS_TAG} {args.system}\n"
    print("Chat started. Type 'exit' to quit.")
    while True:
        user = input("User: ").strip()
        if user.lower() in {"exit", "quit"}:
            break
        prompt = history + f"{USER_TAG} {user}\n{ASSIST_TAG} "
        ids = tokenizer.encode(prompt).ids
        x = torch.tensor([ids], dtype=torch.long, device=args.device)
        with torch.no_grad():
            y = model.generate(x, max_new_tokens=args.max_new_tokens, temperature=args.temperature, top_p=args.top_p, eos_id=eos_id)
        text = tokenizer.decode(y[0].tolist())
        # extract only the latest assistant chunk
        if ASSIST_TAG in text:
            resp = text.split(f"{ASSIST_TAG} ")[-1]
            if EOS_TAG in resp:
                resp = resp.split(EOS_TAG)[0]
        else:
            resp = text
        print(f"Assistant: {resp.strip()}\n")
        history += f"{USER_TAG} {user}\n{ASSIST_TAG} {resp.strip()} {EOS_TAG}\n"

if __name__ == "__main__":
    main()