import argparse
import csv
import statistics
import time

import torch
from transformers import AutoModelForCausalLM

from load_hf_gpt2 import copy_hf_weights
from model_gpt2 import GPT2, config_from_hf


def parse_int_list(value):
    return [
        int(item.strip())
        for item in value.split(",")
        if item.strip()
    ]


def sync_device(device):
    if device == "cuda":
        torch.cuda.synchronize()


def load_local_gpt2(model_name, device, local_files_only):
    hf_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        local_files_only=local_files_only
    )
    hf_model.eval()

    local_model = GPT2(
        config_from_hf(hf_model.config)
    )

    copy_hf_weights(
        hf_model,
        local_model
    )

    local_model.eval().to(device)

    return local_model


def make_random_input(model, batch_size, prompt_len, device):
    return torch.randint(
        0,
        model.config.vocab_size,
        (
            batch_size,
            prompt_len
        ),
        dtype=torch.long,
        device=device
    )


def time_generate(
    model,
    idx,
    max_new_tokens,
    use_cache,
    repeats,
    warmup,
    device
):
    generate_fn = model.generate_kv if use_cache else model.generate

    for _ in range(warmup):
        _ = generate_fn(
            idx.clone(),
            max_new_tokens=max_new_tokens,
            greedy=True
        )

    sync_device(device)

    times = []
    peak_memory_mb = None
    peak_memory_delta_mb = None
    output = None

    for _ in range(repeats):
        baseline_memory_mb = None

        if device == "cuda":
            baseline_memory_mb = torch.cuda.memory_allocated() / (1024 ** 2)
            torch.cuda.reset_peak_memory_stats()

        sync_device(device)
        start = time.perf_counter()
        output = generate_fn(
            idx.clone(),
            max_new_tokens=max_new_tokens,
            greedy=True
        )
        sync_device(device)
        times.append(time.perf_counter() - start)

        if device == "cuda":
            current_peak = torch.cuda.max_memory_allocated() / (1024 ** 2)
            current_delta = current_peak - baseline_memory_mb
            peak_memory_mb = (
                current_peak
                if peak_memory_mb is None
                else max(peak_memory_mb, current_peak)
            )
            peak_memory_delta_mb = (
                current_delta
                if peak_memory_delta_mb is None
                else max(peak_memory_delta_mb, current_delta)
            )

    latency_s = sum(times) / len(times)
    tokens = idx.shape[0] * max_new_tokens

    return {
        "latency_s": latency_s,
        "median_latency_s": statistics.median(times),
        "min_latency_s": min(times),
        "max_latency_s": max(times),
        "tokens_per_s": tokens / latency_s,
        "peak_memory_mb": peak_memory_mb,
        "peak_memory_delta_mb": peak_memory_delta_mb,
        "output_shape": tuple(output.shape),
    }


def append_pair(
    rows,
    suite,
    model,
    batch_size,
    prompt_len,
    max_new_tokens,
    repeats,
    warmup,
    device
):
    torch.manual_seed(42)
    idx = make_random_input(
        model,
        batch_size,
        prompt_len,
        device
    )

    no_cache = time_generate(
        model,
        idx,
        max_new_tokens,
        use_cache=False,
        repeats=repeats,
        warmup=warmup,
        device=device
    )

    kv_cache = time_generate(
        model,
        idx,
        max_new_tokens,
        use_cache=True,
        repeats=repeats,
        warmup=warmup,
        device=device
    )

    speedup = no_cache["latency_s"] / kv_cache["latency_s"]

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
                "total_seq_len": prompt_len + max_new_tokens,
                "latency_s": result["latency_s"],
                "median_latency_s": result["median_latency_s"],
                "min_latency_s": result["min_latency_s"],
                "max_latency_s": result["max_latency_s"],
                "tokens_per_s": result["tokens_per_s"],
                "peak_memory_mb": result["peak_memory_mb"],
                "peak_memory_delta_mb": result["peak_memory_delta_mb"],
                "speedup_vs_no_cache": speedup if method == "kv_cache" else 1.0,
                "output_shape": result["output_shape"],
            }
        )

    return no_cache, kv_cache, speedup


def print_pair(title, no_cache, kv_cache, speedup):
    print(title)
    print(
        f"  no_cache: {no_cache['latency_s']:.4f}s, "
        f"{no_cache['tokens_per_s']:.2f} tokens/s, "
        f"delta {no_cache['peak_memory_delta_mb']:.1f} MB"
    )
    print(
        f"  kv_cache: {kv_cache['latency_s']:.4f}s, "
        f"{kv_cache['tokens_per_s']:.2f} tokens/s, "
        f"delta {kv_cache['peak_memory_delta_mb']:.1f} MB, "
        f"speedup {speedup:.2f}x"
    )


def write_csv(path, rows):
    with open(
        path,
        "w",
        newline="",
        encoding="utf-8"
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(rows[0].keys())
        )
        writer.writeheader()
        writer.writerows(rows)


def check_consistency(model, device):
    torch.manual_seed(123)
    idx = make_random_input(
        model,
        batch_size=2,
        prompt_len=16,
        device=device
    )

    no_cache = model.generate(
        idx.clone(),
        max_new_tokens=16,
        greedy=True
    )

    kv_cache = model.generate_kv(
        idx.clone(),
        max_new_tokens=16,
        greedy=True
    )

    return torch.equal(
        no_cache,
        kv_cache
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        default="gpt2"
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument(
        "--out",
        default="benchmark_gpt2.csv"
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=2
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=1
    )
    parser.add_argument(
        "--generated-lengths",
        default="32,64,128,256"
    )
    parser.add_argument(
        "--prompt-lengths",
        default="64,128,256,512"
    )
    parser.add_argument(
        "--batch-sizes",
        default="1,2,4,8"
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
        "--allow-download",
        action="store_true",
        help="Allow downloading the HuggingFace model if it is not cached."
    )

    args = parser.parse_args()

    model = load_local_gpt2(
        args.model,
        args.device,
        local_files_only=not args.allow_download
    )

    print("=" * 80)
    print("GPT-2 benchmark: generate() vs generate_kv()")
    print("=" * 80)
    print(f"model: {args.model}")
    print(f"device: {args.device}")
    print(f"block_size: {model.config.block_size}")
    print(f"n_layer: {model.config.n_layer}")
    print(f"n_head: {model.config.n_head}")
    print(f"n_embd: {model.config.n_embd}")
    print(f"repeats: {args.repeats}, warmup: {args.warmup}")
    print(f"greedy consistency check: {'PASS' if check_consistency(model, args.device) else 'FAIL'}")
    print()

    rows = []

    print("Suite 1: generated length latency")
    for new_tokens in parse_int_list(args.generated_lengths):
        no_cache, kv_cache, speedup = append_pair(
            rows,
            "generated_length_latency",
            model,
            batch_size=1,
            prompt_len=args.fixed_prompt_len,
            max_new_tokens=new_tokens,
            repeats=args.repeats,
            warmup=args.warmup,
            device=args.device
        )
        print_pair(
            f"  new_tokens={new_tokens}",
            no_cache,
            kv_cache,
            speedup
        )

    print()
    print("Suite 2: prompt length latency")
    for prompt_len in parse_int_list(args.prompt_lengths):
        no_cache, kv_cache, speedup = append_pair(
            rows,
            "prompt_length_latency",
            model,
            batch_size=1,
            prompt_len=prompt_len,
            max_new_tokens=args.fixed_new_tokens,
            repeats=args.repeats,
            warmup=args.warmup,
            device=args.device
        )
        print_pair(
            f"  prompt_len={prompt_len}",
            no_cache,
            kv_cache,
            speedup
        )

    print()
    print("Suite 3: batch size throughput")
    for batch_size in parse_int_list(args.batch_sizes):
        try:
            no_cache, kv_cache, speedup = append_pair(
                rows,
                "batch_size_throughput",
                model,
                batch_size=batch_size,
                prompt_len=args.fixed_prompt_len,
                max_new_tokens=args.fixed_new_tokens,
                repeats=args.repeats,
                warmup=args.warmup,
                device=args.device
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

    write_csv(
        args.out,
        rows
    )

    print()
    print("=" * 80)
    print(f"Saved CSV: {args.out}")
    print("=" * 80)


if __name__ == "__main__":
    main()
