import argparse
import csv
import os
import statistics
import time

import torch

from config import block_size, device, n_head
from model import GPT as GPTNoCache
from model_kvcache import GPT as GPTKVCache

"""
Benchmark no-cache generation against the KV-cache implementation.

The no-cache baseline is loaded from model.py. The KV-cache model is loaded
from model_kvcache.py. Existing checkpoints were trained with the original
per-head attention module, so this script converts those checkpoint keys into
the vectorized KV-cache model format before benchmarking.
"""


def parse_int_list(value):
    return [
        int(item.strip())
        for item in value.split(",")
        if item.strip()
    ]


def sync_device():
    if device == "cuda":
        torch.cuda.synchronize()


def load_models(checkpoint_path):
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"Missing checkpoint: {checkpoint_path}. Run trained.py first."
        )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False
    )

    no_cache_model = GPTNoCache(checkpoint["vocab_size"])
    no_cache_model.load_state_dict(checkpoint["model_state_dict"])
    no_cache_model = no_cache_model.to(device)
    no_cache_model.eval()

    kv_cache_model = GPTKVCache(checkpoint["vocab_size"])
    kv_cache_model.load_state_dict(
        convert_state_dict_for_vectorized_kv(
            checkpoint["model_state_dict"]
        )
    )
    kv_cache_model = kv_cache_model.to(device)
    kv_cache_model.eval()

    return no_cache_model, kv_cache_model, checkpoint["vocab_size"]


def convert_state_dict_for_vectorized_kv(state_dict):
    converted = {}
    consumed = set()

    for key, value in state_dict.items():
        if ".sa.heads." in key:
            continue

        converted[key] = value

    layer_ids = sorted(
        {
            int(key.split(".")[1])
            for key in state_dict
            if key.startswith("blocks.")
            and ".sa.heads." in key
            and key.endswith(".key.weight")
        }
    )

    for layer_id in layer_ids:
        for name in ["key", "query", "value"]:
            pieces = []

            for head_id in range(n_head):
                old_key = (
                    f"blocks.{layer_id}.sa.heads."
                    f"{head_id}.{name}.weight"
                )

                pieces.append(state_dict[old_key])
                consumed.add(old_key)

            converted[
                f"blocks.{layer_id}.sa.{name}.weight"
            ] = torch.cat(
                pieces,
                dim=0
            )

    return converted


def time_generation(
    model,
    idx,
    max_new_tokens,
    use_cache,
    repeats,
    warmup,
    greedy=True,
    temperature=1.0,
    top_k=None,
    top_p=None
):
    generate_fn = model.generate_kv if use_cache else model.generate

    for _ in range(warmup):
        torch.manual_seed(1234)
        _ = generate_fn(
            idx.clone(),
            max_new_tokens,
            greedy=greedy,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p
        )

    sync_device()

    times = []
    last_output = None
    peak_memory_mb = None
    peak_memory_delta_mb = None

    for _ in range(repeats):
        torch.manual_seed(1234)
        idx_run = idx.clone()

        baseline_memory_mb = None

        if device == "cuda":
            baseline_memory_mb = torch.cuda.memory_allocated() / (1024 ** 2)
            torch.cuda.reset_peak_memory_stats()

        sync_device()
        start = time.perf_counter()
        last_output = generate_fn(
            idx_run,
            max_new_tokens,
            greedy=greedy,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p
        )
        sync_device()
        times.append(time.perf_counter() - start)

        if device == "cuda":
            current_peak = torch.cuda.max_memory_allocated() / (1024 ** 2)
            current_delta = current_peak - baseline_memory_mb

            if peak_memory_mb is None:
                peak_memory_mb = current_peak
            else:
                peak_memory_mb = max(peak_memory_mb, current_peak)

            if peak_memory_delta_mb is None:
                peak_memory_delta_mb = current_delta
            else:
                peak_memory_delta_mb = max(peak_memory_delta_mb, current_delta)

    latency_s = sum(times) / len(times)
    median_latency_s = statistics.median(times)
    min_latency_s = min(times)
    max_latency_s = max(times)
    batch_size = idx.shape[0]
    generated_tokens = batch_size * max_new_tokens
    throughput = generated_tokens / latency_s

    return {
        "latency_s": latency_s,
        "median_latency_s": median_latency_s,
        "min_latency_s": min_latency_s,
        "max_latency_s": max_latency_s,
        "tokens_per_s": throughput,
        "peak_memory_mb": peak_memory_mb,
        "peak_memory_delta_mb": peak_memory_delta_mb,
        "output_shape": tuple(last_output.shape),
    }


def make_random_idx(vocab_size, batch_size, prompt_len):
    return torch.randint(
        0,
        vocab_size,
        (batch_size, prompt_len),
        dtype=torch.long,
        device=device
    )


def append_comparison(
    rows,
    suite,
    vocab_size,
    model,
    kv_model,
    batch_size,
    prompt_len,
    max_new_tokens,
    repeats,
    warmup,
    greedy,
    temperature,
    top_k,
    top_p
):
    torch.manual_seed(42)
    idx = make_random_idx(
        vocab_size,
        batch_size,
        prompt_len
    )

    no_cache = time_generation(
        model,
        idx,
        max_new_tokens,
        use_cache=False,
        repeats=repeats,
        warmup=warmup,
        greedy=greedy,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p
    )

    kv_cache = time_generation(
        kv_model,
        idx,
        max_new_tokens,
        use_cache=True,
        repeats=repeats,
        warmup=warmup,
        greedy=greedy,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p
    )

    speedup = no_cache["latency_s"] / kv_cache["latency_s"]
    total_seq_len = prompt_len + max_new_tokens

    for method, result in [
        ("no_cache", no_cache),
        ("kv_cache", kv_cache)
    ]:
        rows.append(
            {
                "suite": suite,
                "method": method,
                "batch_size": batch_size,
                "prompt_len": prompt_len,
                "max_new_tokens": max_new_tokens,
                "total_seq_len": total_seq_len,
                "effective_context_len": min(total_seq_len, block_size),
                "block_size": block_size,
                "latency_s": result["latency_s"],
                "median_latency_s": result["median_latency_s"],
                "min_latency_s": result["min_latency_s"],
                "max_latency_s": result["max_latency_s"],
                "tokens_per_s": result["tokens_per_s"],
                "peak_memory_mb": result["peak_memory_mb"],
                "peak_memory_delta_mb": result["peak_memory_delta_mb"],
                "speedup_vs_no_cache": speedup if method == "kv_cache" else 1.0,
                "output_shape": result["output_shape"],
                "note": (
                    "total length exceeds block_size; current model uses a "
                    "rebuilt block_size window"
                    if total_seq_len > block_size
                    else ""
                )
            }
        )

    return no_cache, kv_cache, speedup


def print_pair(title, no_cache, kv_cache, speedup):
    print(title)
    print(
        f"  no_cache: {no_cache['latency_s']:.4f}s, "
        f"median {no_cache['median_latency_s']:.4f}s, "
        f"{no_cache['tokens_per_s']:.2f} tokens/s, "
        f"peak {no_cache['peak_memory_mb']:.1f} MB, "
        f"delta {no_cache['peak_memory_delta_mb']:.1f} MB"
    )
    print(
        f"  kv_cache: {kv_cache['latency_s']:.4f}s, "
        f"median {kv_cache['median_latency_s']:.4f}s, "
        f"{kv_cache['tokens_per_s']:.2f} tokens/s, "
        f"peak {kv_cache['peak_memory_mb']:.1f} MB, "
        f"delta {kv_cache['peak_memory_delta_mb']:.1f} MB, "
        f"speedup {speedup:.2f}x"
    )


def write_csv(path, rows):
    if not rows:
        return

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(rows[0].keys())
        )
        writer.writeheader()
        writer.writerows(rows)


def check_greedy_consistency(no_cache_model, kv_cache_model, vocab_size, max_new_tokens):
    torch.manual_seed(7)
    idx = make_random_idx(
        vocab_size,
        batch_size=2,
        prompt_len=min(16, block_size)
    )

    out_no_cache = no_cache_model.generate(
        idx.clone(),
        max_new_tokens,
        greedy=True
    )

    out_kv_cache = kv_cache_model.generate_kv(
        idx.clone(),
        max_new_tokens,
        greedy=True
    )

    return torch.equal(out_no_cache, out_kv_cache)


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark current model_kvcache.py implementation."
    )
    parser.add_argument(
        "--checkpoint",
        default="gpt_model.pt"
    )
    parser.add_argument(
        "--out",
        default="benchmark_results.csv"
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=3
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=1
    )
    parser.add_argument(
        "--generated-lengths",
        default="64,128,256,512,1024"
    )
    parser.add_argument(
        "--prompt-lengths",
        default="64,128,256,512,1024"
    )
    parser.add_argument(
        "--batch-sizes",
        default="1,2,4,8,16"
    )
    parser.add_argument(
        "--fixed-prompt-len",
        type=int,
        default=64
    )
    parser.add_argument(
        "--fixed-new-tokens",
        type=int,
        default=64
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=None
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=None
    )
    parser.add_argument(
        "--sample",
        action="store_true",
        help="Use stochastic sampling instead of greedy decoding."
    )

    args = parser.parse_args()

    if args.repeats <= 0:
        raise ValueError("--repeats must be positive")

    no_cache_model, kv_cache_model, vocab_size = load_models(args.checkpoint)
    generated_lengths = parse_int_list(args.generated_lengths)
    prompt_lengths = parse_int_list(args.prompt_lengths)
    batch_sizes = parse_int_list(args.batch_sizes)

    greedy = not args.sample

    print("=" * 80)
    print("GPT benchmark: no_cache generate() vs kv_cache generate_kv()")
    print("=" * 80)
    print(f"device: {device}")
    print(f"vocab_size: {vocab_size}")
    print(f"block_size: {block_size}")
    print(f"repeats: {args.repeats}, warmup: {args.warmup}")
    print(
        "decoding: "
        + (
            f"sampling temperature={args.temperature}, "
            f"top_k={args.top_k}, top_p={args.top_p}"
            if args.sample
            else "greedy"
        )
    )
    print()

    consistent = check_greedy_consistency(
        no_cache_model,
        kv_cache_model,
        vocab_size,
        max_new_tokens=min(32, args.fixed_new_tokens)
    )
    print(f"greedy consistency check: {'PASS' if consistent else 'FAIL'}")
    print()

    rows = []

    print("Suite 0: one-step latency by sequence length")
    print("  This directly covers sequence length 64 -> 1024.")
    for sequence_len in prompt_lengths:
        prompt_len = min(sequence_len, block_size)
        no_cache, kv_cache, speedup = append_comparison(
            rows,
            "sequence_length_one_step_latency",
            vocab_size,
            no_cache_model,
            kv_cache_model,
            batch_size=1,
            prompt_len=prompt_len,
            max_new_tokens=1,
            repeats=args.repeats,
            warmup=args.warmup,
            greedy=greedy,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p
        )
        print_pair(
            f"  sequence_len={sequence_len}, prompt_len={prompt_len}, new_tokens=1",
            no_cache,
            kv_cache,
            speedup
        )

    print()
    print("Suite 1: latency curve by generated length")
    print(
        f"  fixed prompt_len={args.fixed_prompt_len}, "
        "batch_size=1"
    )
    for max_new_tokens in generated_lengths:
        no_cache, kv_cache, speedup = append_comparison(
            rows,
            "generated_length_latency",
            vocab_size,
            no_cache_model,
            kv_cache_model,
            batch_size=1,
            prompt_len=min(args.fixed_prompt_len, block_size),
            max_new_tokens=max_new_tokens,
            repeats=args.repeats,
            warmup=args.warmup,
            greedy=greedy,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p
        )
        print_pair(
            f"  new_tokens={max_new_tokens}",
            no_cache,
            kv_cache,
            speedup
        )

    print()
    print("Suite 2: latency curve by prompt/context length")
    print(
        f"  fixed new_tokens={args.fixed_new_tokens}, "
        "batch_size=1"
    )
    for prompt_len in prompt_lengths:
        effective_prompt_len = min(prompt_len, block_size)
        no_cache, kv_cache, speedup = append_comparison(
            rows,
            "prompt_length_latency",
            vocab_size,
            no_cache_model,
            kv_cache_model,
            batch_size=1,
            prompt_len=effective_prompt_len,
            max_new_tokens=args.fixed_new_tokens,
            repeats=args.repeats,
            warmup=args.warmup,
            greedy=greedy,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p
        )
        suffix = (
            f" requested_prompt_len={prompt_len},"
            if prompt_len != effective_prompt_len
            else ""
        )
        print_pair(
            f"  {suffix} prompt_len={effective_prompt_len}",
            no_cache,
            kv_cache,
            speedup
        )

    print()
    print("Suite 3: throughput by batch size")
    print(
        f"  fixed prompt_len={min(args.fixed_prompt_len, block_size)}, "
        f"new_tokens={args.fixed_new_tokens}"
    )
    for batch_size in batch_sizes:
        try:
            no_cache, kv_cache, speedup = append_comparison(
                rows,
                "batch_size_throughput",
                vocab_size,
                no_cache_model,
                kv_cache_model,
                batch_size=batch_size,
                prompt_len=min(args.fixed_prompt_len, block_size),
                max_new_tokens=args.fixed_new_tokens,
                repeats=args.repeats,
                warmup=args.warmup,
                greedy=greedy,
                temperature=args.temperature,
                top_k=args.top_k,
                top_p=args.top_p
            )
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            print(f"  batch_size={batch_size}: CUDA OOM, skipped")
            continue

        print_pair(
            f"  batch_size={batch_size}",
            no_cache,
            kv_cache,
            speedup
        )

    write_csv(args.out, rows)

    print()
    print("=" * 80)
    print(f"Saved CSV: {args.out}")
    print(
        "Columns include latency_s, median_latency_s, min_latency_s, "
        "max_latency_s, tokens_per_s, peak_memory_mb, "
        "peak_memory_delta_mb, speedup_vs_no_cache, total_seq_len, "
        "effective_context_len, and block_size."
    )
    if any(row["total_seq_len"] > block_size for row in rows):
        print(
            "Note: rows with total_seq_len > block_size use the current "
            "model's rebuilt block_size window, not unlimited context."
        )
    print("=" * 80)


if __name__ == "__main__":
    main()
