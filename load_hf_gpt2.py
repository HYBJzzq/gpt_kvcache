import argparse

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from model_gpt2 import GPT2, config_from_hf


TRANSPOSED = (
    "attn.c_attn.weight",
    "attn.c_proj.weight",
    "mlp.c_fc.weight",
    "mlp.c_proj.weight",
)


def copy_hf_weights(hf_model, local_model):
    hf_state = hf_model.state_dict()
    local_state = local_model.state_dict()

    copied = {}

    for key in local_state:
        if key == "lm_head.weight":
            copied[key] = hf_state["transformer.wte.weight"]
            continue

        if key not in hf_state:
            raise KeyError(f"missing HuggingFace key: {key}")

        value = hf_state[key]

        if key.endswith(TRANSPOSED):
            value = value.t()

        if value.shape != local_state[key].shape:
            raise ValueError(
                f"shape mismatch for {key}: "
                f"hf={tuple(value.shape)}, local={tuple(local_state[key].shape)}"
            )

        copied[key] = value

    local_model.load_state_dict(copied)


@torch.no_grad()
def verify_logits(
    model_name,
    prompt,
    device,
    atol
):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    hf_model = AutoModelForCausalLM.from_pretrained(model_name)
    hf_model.eval().to(device)

    local_model = GPT2(
        config_from_hf(hf_model.config)
    )
    copy_hf_weights(
        hf_model,
        local_model
    )
    local_model.eval().to(device)

    encoded = tokenizer(
        prompt,
        return_tensors="pt"
    )

    input_ids = encoded["input_ids"].to(device)

    hf_logits = hf_model(
        input_ids
    ).logits

    local_logits = local_model(
        input_ids
    )

    max_abs_error = (
        hf_logits - local_logits
    ).abs().max().item()

    mean_abs_error = (
        hf_logits - local_logits
    ).abs().mean().item()

    same_argmax = torch.equal(
        hf_logits.argmax(dim=-1),
        local_logits.argmax(dim=-1)
    )

    print(f"model: {model_name}")
    print(f"device: {device}")
    print(f"prompt: {prompt!r}")
    print(f"input shape: {tuple(input_ids.shape)}")
    print(f"max_abs_error: {max_abs_error:.8f}")
    print(f"mean_abs_error: {mean_abs_error:.8f}")
    print(f"argmax_equal: {same_argmax}")
    print(f"pass_atol_{atol}: {max_abs_error < atol}")

    hf_prefill = hf_model(
        input_ids,
        use_cache=True
    )

    local_past = local_model.init_kv_cache(
        input_ids.shape[0],
        local_model.config.block_size,
        input_ids.device
    )

    local_prefill_logits, local_past, local_cache_len = local_model(
        input_ids,
        past_key_values=local_past,
        cache_len=0,
        use_cache=True
    )

    prefill_error = (
        hf_prefill.logits - local_prefill_logits
    ).abs().max().item()

    next_token = hf_prefill.logits[
        :,
        -1,
        :
    ].argmax(
        dim=-1,
        keepdim=True
    )

    hf_decode = hf_model(
        next_token,
        past_key_values=hf_prefill.past_key_values,
        use_cache=True
    )

    local_decode_logits, _, _ = local_model(
        next_token,
        past_key_values=local_past,
        cache_len=local_cache_len,
        use_cache=True
    )

    decode_error = (
        hf_decode.logits - local_decode_logits
    ).abs().max().item()

    local_generate = local_model.generate(
        input_ids.clone(),
        max_new_tokens=8,
        greedy=True
    )

    local_generate_kv = local_model.generate_kv(
        input_ids.clone(),
        max_new_tokens=8,
        greedy=True
    )

    generate_equal = torch.equal(
        local_generate,
        local_generate_kv
    )

    print(f"prefill_cache_max_abs_error: {prefill_error:.8f}")
    print(f"decode_cache_max_abs_error: {decode_error:.8f}")
    print(f"local_generate_equal_kv: {generate_equal}")

    return {
        "max_abs_error": max_abs_error,
        "mean_abs_error": mean_abs_error,
        "argmax_equal": same_argmax,
        "prefill_cache_max_abs_error": prefill_error,
        "decode_cache_max_abs_error": decode_error,
        "local_generate_equal_kv": generate_equal,
        "passed": max_abs_error < atol
        and prefill_error < atol
        and decode_error < atol
        and generate_equal,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        default="gpt2"
    )
    parser.add_argument(
        "--prompt",
        default="Hello, my name is"
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument(
        "--atol",
        type=float,
        default=1e-4
    )

    args = parser.parse_args()

    verify_logits(
        args.model,
        args.prompt,
        args.device,
        args.atol
    )


if __name__ == "__main__":
    main()
