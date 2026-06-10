import argparse

import torch

from benchmark import load_models
from config import device
from load_hf_gpt2 import verify_logits
from model_kvcache import pad_prompts


def check_char_kv(checkpoint):
    no_cache_model, kv_cache_model, vocab_size = load_models(checkpoint)

    torch.manual_seed(1234)
    idx = torch.randint(
        0,
        vocab_size,
        (2, 16),
        device=device
    )

    no_cache = no_cache_model.generate(
        idx.clone(),
        max_new_tokens=16,
        greedy=True
    )

    kv_cache = kv_cache_model.generate_kv(
        idx.clone(),
        max_new_tokens=16,
        greedy=True
    )

    ok = torch.equal(
        no_cache,
        kv_cache
    )

    print(f"char_generate_equal_kv: {ok}")
    return ok


def check_static_batching(checkpoint):
    _, kv_cache_model, _ = load_models(checkpoint)

    prompts = [
        [1, 2, 3, 4, 5],
        [6, 7],
        [8, 9, 10],
    ]

    input_ids, attention_mask = pad_prompts(
        prompts,
        device=device
    )

    batched = kv_cache_model.generate_kv_batch(
        input_ids,
        attention_mask,
        max_new_tokens=6,
        greedy=True
    )

    max_prompt_len = input_ids.shape[1]
    ok = True

    for i, prompt in enumerate(prompts):
        single = kv_cache_model.generate_kv(
            torch.tensor(
                [prompt],
                device=device
            ),
            max_new_tokens=6,
            greedy=True
        )[0, len(prompt):]

        generated = batched[
            i,
            max_prompt_len:max_prompt_len + 6
        ]

        same = torch.equal(
            single,
            generated
        )

        print(f"static_batch_row_{i}: {same}")
        ok = ok and same

    print(f"static_batching_equal_single: {ok}")
    return ok


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        default="gpt_model.pt"
    )
    parser.add_argument(
        "--skip-hf",
        action="store_true",
        help="Skip HuggingFace GPT-2 verification."
    )

    args = parser.parse_args()

    checks = [
        check_char_kv(args.checkpoint),
        check_static_batching(args.checkpoint),
    ]

    if not args.skip_hf:
        result = verify_logits(
            "gpt2",
            "Hello, my name is",
            device,
            1e-4
        )
        checks.append(result["passed"])

    passed = all(checks)
    print(f"all_checks_passed: {passed}")

    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
