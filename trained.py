import argparse
import os

import torch
from tqdm import tqdm

from config import *
from model import GPT

"""
Train or load the char-level GPT baseline.

Training uses model.py on purpose. The KV-cache implementation is an
inference experiment in model_kvcache.py and should not be required for
checkpoint creation.
"""


torch.manual_seed(1337)


with open("input.txt", "r", encoding="utf-8") as f:
    text = f.read()

chars = sorted(list(set(text)))
vocab_size = len(chars)

stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}


def encode(s):
    return [stoi[c] for c in s]


def decode(tokens):
    return "".join([itos[i] for i in tokens])


data = torch.tensor(
    encode(text),
    dtype=torch.long
)

n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]


def get_batch(split):
    source = train_data if split == "train" else val_data

    if len(source) <= block_size:
        raise ValueError(
            f"{split} data length must be greater than block_size={block_size}"
        )

    ix = torch.randint(
        len(source) - block_size,
        (batch_size,)
    )

    x = torch.stack(
        [source[i:i + block_size] for i in ix]
    )

    y = torch.stack(
        [source[i + 1:i + block_size + 1] for i in ix]
    )

    return x.to(device), y.to(device)


@torch.no_grad()
def estimate_loss(model):
    out = {}
    model.eval()

    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters)

        for k in range(eval_iters):
            x, y = get_batch(split)
            _, loss = model(x, y)
            losses[k] = loss.item()

        out[split] = losses.mean().item()

    model.train()
    return out


def checkpoint_config():
    return {
        "block_size": block_size,
        "batch_size": batch_size,
        "max_iters": max_iters,
        "eval_interval": eval_interval,
        "learning_rate": learning_rate,
        "eval_iters": eval_iters,
        "n_embd": n_embd,
        "n_head": n_head,
        "n_layer": n_layer,
        "dropout": dropout,
    }


def save_model(model, path="gpt_model.pt"):
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "vocab_size": vocab_size,
            "config": checkpoint_config(),
            "chars": chars,
        },
        path
    )
    print(f"Saved model to {path}")


def load_model(path="gpt_model.pt"):
    checkpoint = torch.load(
        path,
        map_location=device,
        weights_only=False
    )

    saved_config = checkpoint.get("config")
    if saved_config is not None:
        current_config = checkpoint_config()
        mismatches = [
            key
            for key in [
                "block_size",
                "n_embd",
                "n_head",
                "n_layer",
            ]
            if saved_config.get(key) != current_config.get(key)
        ]

        if mismatches:
            mismatch_text = ", ".join(
                f"{key}: checkpoint={saved_config.get(key)}, current={current_config.get(key)}"
                for key in mismatches
            )
            raise ValueError(
                "Checkpoint architecture does not match current config: "
                + mismatch_text
            )

    model = GPT(checkpoint["vocab_size"])
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()
    print(f"Loaded model from {path}")
    return model


def train(args):
    model = GPT(vocab_size).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate
    )

    pbar = tqdm(
        range(max_iters),
        desc="training",
        unit="step",
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] {postfix}"
    )

    for step in pbar:
        xb, yb = get_batch("train")

        _, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if step % 100 == 0:
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        if args.eval and step > 0 and step % eval_interval == 0:
            losses = estimate_loss(model)
            pbar.set_postfix(
                {
                    "train": f"{losses['train']:.4f}",
                    "val": f"{losses['val']:.4f}",
                }
            )

    save_model(model, args.checkpoint)
    return model


def smoke_generate(model, max_new_tokens):
    model.eval()

    context = torch.zeros(
        (1, 1),
        dtype=torch.long,
        device=device
    )

    output = model.generate(
        context,
        max_new_tokens=max_new_tokens,
        temperature=0.8,
        top_k=50
    )

    print(decode(output[0].tolist()))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        default="gpt_model.pt"
    )
    parser.add_argument(
        "--force-train",
        action="store_true",
        help="Train even when a checkpoint already exists."
    )
    parser.add_argument(
        "--eval",
        action="store_true",
        help="Periodically estimate train/val loss."
    )
    parser.add_argument(
        "--generate-tokens",
        type=int,
        default=100
    )

    args = parser.parse_args()

    print(f"device: {device}")
    print(
        "config: "
        f"block_size={block_size}, batch_size={batch_size}, "
        f"n_embd={n_embd}, n_head={n_head}, n_layer={n_layer}"
    )

    if os.path.exists(args.checkpoint) and not args.force_train:
        model = load_model(args.checkpoint)
    else:
        model = train(args)

    print("sample generation:")
    smoke_generate(
        model,
        args.generate_tokens
    )


if __name__ == "__main__":
    main()
