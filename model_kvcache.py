import torch
import torch.nn as nn
import torch.nn.functional as F

from config import *

"""
Char-level GPT used for inference experiments.

This file is the optimized experiment model:
- vectorized multi-head causal self-attention
- KV cache generation
- greedy / temperature / top-k / top-p sampling
- simple static batching with right padding and attention masks

The training baseline without KV cache lives in model.py.
"""


def pad_prompts(
    prompts,
    pad_token_id=0,
    device=None
):
    max_len = max(
        len(prompt)
        for prompt in prompts
    )

    input_ids = torch.full(
        (
            len(prompts),
            max_len
        ),
        pad_token_id,
        dtype=torch.long,
        device=device
    )

    attention_mask = torch.zeros(
        (
            len(prompts),
            max_len
        ),
        dtype=torch.long,
        device=device
    )

    for i, prompt in enumerate(prompts):
        prompt_tensor = torch.tensor(
            prompt,
            dtype=torch.long,
            device=device
        )

        input_ids[
            i,
            :len(prompt)
        ] = prompt_tensor

        attention_mask[
            i,
            :len(prompt)
        ] = 1

    return input_ids, attention_mask


class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        num_head,
        head_size
    ):
        super().__init__()

        self.num_head = num_head
        self.head_size = head_size

        self.key = nn.Linear(
            n_embd,
            n_embd,
            bias=False
        )

        self.query = nn.Linear(
            n_embd,
            n_embd,
            bias=False
        )

        self.value = nn.Linear(
            n_embd,
            n_embd,
            bias=False
        )

        self.proj = nn.Linear(
            n_embd,
            n_embd
        )

        self.drop = nn.Dropout(
            dropout
        )

    def forward(
        self,
        x,
        past_kv=None,
        cache_len=0,
        cache_write_pos=None,
        new_cache_len=None,
        use_cache=False,
        attention_mask=None
    ):
        B, T, C = x.shape

        q = self.query(x)
        k_new = self.key(x)
        v_new = self.value(x)

        q = q.view(
            B,
            T,
            self.num_head,
            self.head_size
        ).transpose(1, 2)

        k_new = k_new.view(
            B,
            T,
            self.num_head,
            self.head_size
        ).transpose(1, 2)

        v_new = v_new.view(
            B,
            T,
            self.num_head,
            self.head_size
        ).transpose(1, 2)

        if use_cache:
            if past_kv is None:
                raise ValueError(
                    "past_kv must be preallocated when use_cache=True"
                )

            if cache_write_pos is None or new_cache_len is None:
                raise ValueError(
                    "cache_write_pos and new_cache_len are required when use_cache=True"
                )

            cache_k, cache_v = past_kv

            cache_k[
                :,
                :,
                cache_write_pos:cache_write_pos + T,
                :
            ] = k_new

            cache_v[
                :,
                :,
                cache_write_pos:cache_write_pos + T,
                :
            ] = v_new

            k = cache_k[
                :,
                :,
                :new_cache_len,
                :
            ]

            v = cache_v[
                :,
                :,
                :new_cache_len,
                :
            ]

            past_len = cache_write_pos
            kv = past_kv
        else:
            k = k_new
            v = v_new
            past_len = 0
            kv = None

        att = (
            q @ k.transpose(-2, -1)
        ) * (self.head_size ** -0.5)

        total_T = k.shape[-2]

        q_pos = torch.arange(
            past_len,
            past_len + T,
            device=x.device
        ).view(T, 1)

        k_pos = torch.arange(
            total_T,
            device=x.device
        ).view(1, total_T)

        att = att.masked_fill(
            k_pos > q_pos,
            float("-inf")
        )

        if attention_mask is not None:
            att = att.masked_fill(
                attention_mask[
                    :,
                    None,
                    None,
                    :total_T
                ] == 0,
                float("-inf")
            )

        att = F.softmax(
            att,
            dim=-1
        )

        att = self.drop(att)

        x = att @ v

        x = x.transpose(
            1,
            2
        ).contiguous().view(
            B,
            T,
            C
        )

        x = self.proj(x)
        x = self.drop(x)

        return x, kv


class FeedFoward(nn.Module):
    def __init__(self):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(
                n_embd,
                4 * n_embd
            ),
            nn.ReLU(),
            nn.Linear(
                4 * n_embd,
                n_embd
            ),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        x
    ):
        return self.net(x)


class Block(nn.Module):
    def __init__(self):
        super().__init__()

        head_size = (
            n_embd // n_head
        )

        self.sa = MultiHeadAttention(
            n_head,
            head_size
        )

        self.ffwd = FeedFoward()

        self.ln1 = nn.LayerNorm(
            n_embd
        )

        self.ln2 = nn.LayerNorm(
            n_embd
        )

    def forward(
        self,
        x,
        past_kv=None,
        cache_len=0,
        cache_write_pos=None,
        new_cache_len=None,
        use_cache=False,
        attention_mask=None
    ):
        x = self.ln1(x)

        sa_out, kv = self.sa(
            x,
            past_kv,
            cache_len=cache_len,
            cache_write_pos=cache_write_pos,
            new_cache_len=new_cache_len,
            use_cache=use_cache,
            attention_mask=attention_mask
        )

        x = x + sa_out

        x = self.ln2(x)

        x = x + self.ffwd(x)

        return x, kv


class GPT(nn.Module):
    def __init__(
        self,
        vocab_size
    ):
        super().__init__()

        self.token_embedding_table = nn.Embedding(
            vocab_size,
            n_embd
        )

        self.position_embedding_table = nn.Embedding(
            block_size,
            n_embd
        )

        self.blocks = nn.ModuleList(
            [
                Block()
                for _ in range(
                    n_layer
                )
            ]
        )

        self.ln_f = nn.LayerNorm(
            n_embd
        )

        self.lm_head = nn.Linear(
            n_embd,
            vocab_size
        )

    def init_kv_cache(
        self,
        batch_size,
        max_cache_len,
        device
    ):
        head_size = n_embd // n_head
        dtype = next(
            self.parameters()
        ).dtype
        cache = []

        for _ in range(n_layer):
            cache_k = torch.empty(
                batch_size,
                n_head,
                max_cache_len,
                head_size,
                device=device,
                dtype=dtype
            )

            cache_v = torch.empty(
                batch_size,
                n_head,
                max_cache_len,
                head_size,
                device=device,
                dtype=dtype
            )

            cache.append(
                (
                    cache_k,
                    cache_v
                )
            )

        return cache

    def _cache_positions(
        self,
        cache_len,
        token_count,
        max_cache_len=block_size
    ):
        if token_count > max_cache_len:
            raise ValueError(
                f"token_count={token_count} exceeds max_cache_len={max_cache_len}"
            )

        overflow = max(
            0,
            cache_len + token_count - max_cache_len
        )

        if overflow > 0:
            raise ValueError(
                "cache overflow is not supported for learned absolute "
                "position embeddings; rebuild the cache from the latest "
                "block_size tokens instead"
            )

        cache_write_pos = cache_len - overflow
        new_cache_len = min(
            max_cache_len,
            cache_len + token_count
        )

        return cache_write_pos, new_cache_len

    def _sample_next_token(
        self,
        logits,
        temperature=1.0,
        top_k=None,
        top_p=None,
        greedy=False
    ):
        if greedy or temperature == 0:
            return torch.argmax(
                logits,
                dim=-1,
                keepdim=True
            )

        if temperature < 0:
            raise ValueError(
                "temperature must be non-negative"
            )

        logits = logits / temperature

        if top_k is not None:
            if top_k <= 0:
                raise ValueError(
                    "top_k must be positive"
                )

            top_k = min(
                top_k,
                logits.shape[-1]
            )

            values, _ = torch.topk(
                logits,
                top_k,
                dim=-1
            )

            min_values = values[
                :,
                -1
            ].unsqueeze(-1)

            logits = logits.masked_fill(
                logits < min_values,
                float("-inf")
            )

        if top_p is not None:
            if top_p <= 0 or top_p > 1:
                raise ValueError(
                    "top_p must be in the range (0, 1]"
                )

            sorted_logits, sorted_indices = torch.sort(
                logits,
                descending=True,
                dim=-1
            )

            sorted_probs = F.softmax(
                sorted_logits,
                dim=-1
            )

            cumulative_probs = torch.cumsum(
                sorted_probs,
                dim=-1
            )

            sorted_mask = cumulative_probs > top_p

            sorted_mask[
                :,
                1:
            ] = sorted_mask[
                :,
                :-1
            ].clone()

            sorted_mask[
                :,
                0
            ] = False

            sorted_logits = sorted_logits.masked_fill(
                sorted_mask,
                float("-inf")
            )

            filtered_logits = torch.full_like(
                logits,
                float("-inf")
            )

            logits = filtered_logits.scatter(
                dim=-1,
                index=sorted_indices,
                src=sorted_logits
            )

        probs = F.softmax(
            logits,
            dim=-1
        )

        return torch.multinomial(
            probs,
            num_samples=1
        )

    def forward(
        self,
        idx,
        targets=None,
        past_kvs=None,
        cache_len=0,
        use_cache=False,
        attention_mask=None,
        position_ids=None
    ):
        B, T = idx.shape

        if use_cache:
            if past_kvs is None:
                past_kvs = self.init_kv_cache(
                    B,
                    block_size,
                    idx.device
                )

            max_cache_len = past_kvs[0][0].shape[2]

            cache_write_pos, new_cache_len = self._cache_positions(
                cache_len,
                T,
                max_cache_len=max_cache_len
            )

            past_len = cache_write_pos
        else:
            past_len = 0

        if position_ids is None:
            pos = torch.arange(
                past_len,
                past_len + T,
                device=idx.device
            )
        else:
            pos = position_ids

        tok_emb = (
            self.token_embedding_table(
                idx
            )
        )

        pos_emb = (
            self.position_embedding_table(
                pos
            )
        )

        x = tok_emb + pos_emb

        new_kvs = []

        for i, block in enumerate(
            self.blocks
        ):

            layer_cache = None

            if past_kvs is not None:
                layer_cache = (
                    past_kvs[i]
                )

            x, kv = block(
                x,
                layer_cache,
                cache_len=cache_len,
                cache_write_pos=cache_write_pos if use_cache else None,
                new_cache_len=new_cache_len if use_cache else None,
                use_cache=use_cache,
                attention_mask=attention_mask
            )

            new_kvs.append(kv)

        x = self.ln_f(x)

        logits = self.lm_head(x)

        loss = None

        if targets is not None:

            B, T, C = logits.shape

            logits = logits.view(
                B * T,
                C
            )

            targets = targets.view(
                B * T
            )

            loss = (
                F.cross_entropy(
                    logits,
                    targets
                )
            )

        if use_cache:
            return (
                logits,
                loss,
                past_kvs,
                new_cache_len
            )

        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        idx,
        max_new_tokens,
        temperature=1.0,
        top_k=None,
        top_p=None,
        greedy=False
    ):
        for _ in range(
            max_new_tokens
        ):

            idx_cond = idx[
                :,
                -block_size:
            ]

            logits, _ = self(
                idx_cond
            )

            logits = logits[
                :,
                -1,
                :
            ]

            idx_next = self._sample_next_token(
                logits,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                greedy=greedy
            )

            idx = torch.cat(
                (
                    idx,
                    idx_next
                ),
                dim=1
            )

        return idx

    @torch.no_grad()
    def generate_kv(
        self,
        idx,
        max_new_tokens,
        temperature=1.0,
        top_k=None,
        top_p=None,
        greedy=False
    ):
        if max_new_tokens == 0:
            return idx

        idx_cond = idx[
            :,
            -block_size:
        ]

        if idx_cond.shape[1] + max_new_tokens > block_size:
            past_kvs = None
            cache_len = 0

            for _ in range(
                max_new_tokens
            ):
                if past_kvs is None or idx.shape[1] > block_size:
                    idx_cond = idx[
                        :,
                        -block_size:
                    ]

                    past_kvs = self.init_kv_cache(
                        idx_cond.shape[0],
                        block_size,
                        idx_cond.device
                    )

                    logits, _, past_kvs, cache_len = self(
                        idx_cond,
                        past_kvs=past_kvs,
                        cache_len=0,
                        use_cache=True
                    )
                else:
                    logits, _, past_kvs, cache_len = self(
                        idx[
                            :,
                            -1:
                        ],
                        past_kvs=past_kvs,
                        cache_len=cache_len,
                        use_cache=True
                    )

                logits_last = logits[
                    :,
                    -1,
                    :
                ]

                idx_next = self._sample_next_token(
                    logits_last,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                    greedy=greedy
                )

                idx = torch.cat(
                    [
                        idx,
                        idx_next
                    ],
                    dim=1
                )

            return idx

        max_cache_len = min(
            block_size,
            idx_cond.shape[1] + max_new_tokens
        )

        past_kvs = self.init_kv_cache(
            idx.shape[0],
            max_cache_len,
            idx.device
        )

        logits, _, past_kvs, cache_len = self(
            idx_cond,
            past_kvs=past_kvs,
            cache_len=0,
            use_cache=True
        )

        logits_last = logits[
            :,
            -1,
            :
        ]

        idx_next = self._sample_next_token(
            logits_last,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            greedy=greedy
        )

        idx = torch.cat(
            [
                idx,
                idx_next
            ],
            dim=1
        )

        for _ in range(
            max_new_tokens - 1
        ):
            logits, _, past_kvs, cache_len = self(
                idx_next,
                past_kvs=past_kvs,
                cache_len=cache_len,
                use_cache=True
            )

            logits_last = logits[
                :,
                -1,
                :
            ]

            idx_next = self._sample_next_token(
                logits_last,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                greedy=greedy
            )

            idx = torch.cat(
                [
                    idx,
                    idx_next
                ],
                dim=1
            )

        return idx

    @torch.no_grad()
    def generate_kv_batch(
        self,
        input_ids,
        attention_mask,
        max_new_tokens,
        temperature=1.0,
        top_k=None,
        top_p=None,
        greedy=False,
        pad_token_id=0
    ):
        if max_new_tokens == 0:
            return input_ids

        B, T = input_ids.shape
        total_len = T + max_new_tokens

        if total_len > block_size:
            raise ValueError(
                f"static batching requires padded length + new tokens <= block_size; "
                f"got {T} + {max_new_tokens} > {block_size}"
            )

        attention_mask = attention_mask.to(
            device=input_ids.device,
            dtype=torch.long
        )

        lengths = attention_mask.sum(
            dim=1
        )

        if torch.any(lengths == 0):
            raise ValueError(
                "all prompts must contain at least one non-padding token"
            )

        position_ids = attention_mask.cumsum(
            dim=1
        ) - 1

        position_ids = position_ids.masked_fill(
            attention_mask == 0,
            0
        )

        past_kvs = self.init_kv_cache(
            B,
            total_len,
            input_ids.device
        )

        logits, _, past_kvs, cache_len = self(
            input_ids,
            past_kvs=past_kvs,
            cache_len=0,
            use_cache=True,
            attention_mask=attention_mask,
            position_ids=position_ids
        )

        batch_idx = torch.arange(
            B,
            device=input_ids.device
        )

        logits_last = logits[
            batch_idx,
            lengths - 1,
            :
        ]

        idx_next = self._sample_next_token(
            logits_last,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            greedy=greedy
        )

        output_ids = torch.cat(
            [
                input_ids,
                idx_next
            ],
            dim=1
        )

        attention_mask = torch.cat(
            [
                attention_mask,
                torch.ones(
                    B,
                    1,
                    dtype=attention_mask.dtype,
                    device=attention_mask.device
                )
            ],
            dim=1
        )

        generated = 1

        while generated < max_new_tokens:
            next_position_ids = (
                lengths + generated - 1
            ).view(
                B,
                1
            )

            logits, _, past_kvs, cache_len = self(
                idx_next,
                past_kvs=past_kvs,
                cache_len=cache_len,
                use_cache=True,
                attention_mask=attention_mask,
                position_ids=next_position_ids
            )

            logits_last = logits[
                :,
                -1,
                :
            ]

            idx_next = self._sample_next_token(
                logits_last,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                greedy=greedy
            )

            output_ids = torch.cat(
                [
                    output_ids,
                    idx_next
                ],
                dim=1
            )

            attention_mask = torch.cat(
                [
                    attention_mask,
                    torch.ones(
                        B,
                        1,
                        dtype=attention_mask.dtype,
                        device=attention_mask.device
                    )
                ],
                dim=1
            )

            generated += 1

        return output_ids
