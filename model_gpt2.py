import math

import torch
import torch.nn as nn
import torch.nn.functional as F

"""
GPT-2 compatible model.

This implementation is separate from the char-level project model. It is
structured to load HuggingFace GPT-2 weights exactly, then verify both full
forward logits and KV-cache decode logits against HuggingFace.
"""


class GPT2Config:
    def __init__(
        self,
        vocab_size=50257,
        block_size=1024,
        n_layer=12,
        n_head=12,
        n_embd=768,
        dropout=0.0
    ):
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.n_layer = n_layer
        self.n_head = n_head
        self.n_embd = n_embd
        self.dropout = dropout


class GPT2Attention(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_size = config.n_embd // config.n_head

        self.c_attn = nn.Linear(
            config.n_embd,
            3 * config.n_embd
        )

        self.c_proj = nn.Linear(
            config.n_embd,
            config.n_embd
        )

        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        self.register_buffer(
            "bias",
            torch.tril(
                torch.ones(
                    config.block_size,
                    config.block_size
                )
            ).view(
                1,
                1,
                config.block_size,
                config.block_size
            ),
            persistent=False
        )

    def forward(
        self,
        x,
        past_kv=None,
        use_cache=False,
        cache_len=0,
        cache_write_pos=None,
        new_cache_len=None
    ):
        B, T, C = x.shape

        q, k, v = self.c_attn(x).split(
            self.n_embd,
            dim=2
        )

        q = q.view(
            B,
            T,
            self.n_head,
            self.head_size
        ).transpose(1, 2)

        k = k.view(
            B,
            T,
            self.n_head,
            self.head_size
        ).transpose(1, 2)

        v = v.view(
            B,
            T,
            self.n_head,
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
            ] = k

            cache_v[
                :,
                :,
                cache_write_pos:cache_write_pos + T,
                :
            ] = v

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
            present = past_kv
        else:
            past_len = 0
            present = None

        att = (
            q @ k.transpose(-2, -1)
        ) * (1.0 / math.sqrt(self.head_size))

        total_T = k.shape[-2]
        q_pos = torch.arange(
            past_len,
            past_len + T,
            device=x.device
        ).view(
            T,
            1
        )

        k_pos = torch.arange(
            total_T,
            device=x.device
        ).view(
            1,
            total_T
        )

        att = att.masked_fill(
            k_pos > q_pos,
            torch.finfo(att.dtype).min
        )

        att = F.softmax(
            att,
            dim=-1
        )

        att = self.attn_dropout(att)

        y = att @ v

        y = y.transpose(
            1,
            2
        ).contiguous().view(
            B,
            T,
            C
        )

        y = self.c_proj(y)
        y = self.resid_dropout(y)

        return y, present


class GPT2MLP(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.c_fc = nn.Linear(
            config.n_embd,
            4 * config.n_embd
        )

        self.c_proj = nn.Linear(
            4 * config.n_embd,
            config.n_embd
        )

        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = F.gelu(
            x,
            approximate="tanh"
        )
        x = self.c_proj(x)
        x = self.dropout(x)

        return x


class GPT2Block(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.ln_1 = nn.LayerNorm(
            config.n_embd,
            eps=1e-5
        )

        self.attn = GPT2Attention(config)

        self.ln_2 = nn.LayerNorm(
            config.n_embd,
            eps=1e-5
        )

        self.mlp = GPT2MLP(config)

    def forward(
        self,
        x,
        past_kv=None,
        use_cache=False,
        cache_len=0,
        cache_write_pos=None,
        new_cache_len=None
    ):
        attn_out, present = self.attn(
            self.ln_1(x),
            past_kv=past_kv,
            use_cache=use_cache,
            cache_len=cache_len,
            cache_write_pos=cache_write_pos,
            new_cache_len=new_cache_len
        )

        x = x + attn_out

        x = x + self.mlp(
            self.ln_2(x)
        )

        return x, present


class GPT2(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.config = config

        self.transformer = nn.ModuleDict(
            {
                "wte": nn.Embedding(
                    config.vocab_size,
                    config.n_embd
                ),
                "wpe": nn.Embedding(
                    config.block_size,
                    config.n_embd
                ),
                "drop": nn.Dropout(config.dropout),
                "h": nn.ModuleList(
                    [
                        GPT2Block(config)
                        for _ in range(config.n_layer)
                    ]
                ),
                "ln_f": nn.LayerNorm(
                    config.n_embd,
                    eps=1e-5
                ),
            }
        )

        self.lm_head = nn.Linear(
            config.n_embd,
            config.vocab_size,
            bias=False
        )

        self.lm_head.weight = self.transformer["wte"].weight

    def init_kv_cache(
        self,
        batch_size,
        max_cache_len,
        device
    ):
        dtype = next(
            self.parameters()
        ).dtype

        cache = []

        for _ in range(self.config.n_layer):
            cache_k = torch.empty(
                batch_size,
                self.config.n_head,
                max_cache_len,
                self.config.n_embd // self.config.n_head,
                device=device,
                dtype=dtype
            )

            cache_v = torch.empty(
                batch_size,
                self.config.n_head,
                max_cache_len,
                self.config.n_embd // self.config.n_head,
                device=device,
                dtype=dtype
            )

            cache.append(
                (
                    cache_k,
                    cache_v
                )
            )

        return tuple(cache)

    def _cache_positions(
        self,
        cache_len,
        token_count,
        max_cache_len
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

    def forward(
        self,
        idx,
        past_key_values=None,
        use_cache=False,
        cache_len=0
    ):
        B, T = idx.shape

        if use_cache:
            if past_key_values is None:
                past_key_values = self.init_kv_cache(
                    B,
                    self.config.block_size,
                    idx.device
                )

            max_cache_len = past_key_values[0][0].shape[-2]

            cache_write_pos, new_cache_len = self._cache_positions(
                cache_len,
                T,
                max_cache_len=max_cache_len
            )

            past_len = cache_write_pos
        else:
            past_len = 0

        if past_len + T > self.config.block_size:
            raise ValueError(
                f"sequence length {past_len + T} exceeds block_size={self.config.block_size}"
            )

        pos = torch.arange(
            past_len,
            past_len + T,
            dtype=torch.long,
            device=idx.device
        )

        tok_emb = self.transformer["wte"](idx)
        pos_emb = self.transformer["wpe"](pos)

        x = self.transformer["drop"](
            tok_emb + pos_emb
        )

        presents = []

        for i, block in enumerate(self.transformer["h"]):
            past_kv = None

            if past_key_values is not None:
                past_kv = past_key_values[i]

            x, present = block(
                x,
                past_kv=past_kv,
                use_cache=use_cache,
                cache_len=cache_len,
                cache_write_pos=cache_write_pos if use_cache else None,
                new_cache_len=new_cache_len if use_cache else None
            )

            if use_cache:
                presents.append(present)

        x = self.transformer["ln_f"](x)
        logits = self.lm_head(x)

        if use_cache:
            return logits, tuple(presents), new_cache_len

        return logits

    @torch.no_grad()
    def generate(
        self,
        idx,
        max_new_tokens,
        greedy=True
    ):
        for _ in range(max_new_tokens):
            idx_cond = idx[
                :,
                -self.config.block_size:
            ]

            logits = self(idx_cond)
            logits = logits[
                :,
                -1,
                :
            ]

            if greedy:
                idx_next = torch.argmax(
                    logits,
                    dim=-1,
                    keepdim=True
                )
            else:
                probs = F.softmax(
                    logits,
                    dim=-1
                )
                idx_next = torch.multinomial(
                    probs,
                    num_samples=1
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
    def generate_kv(
        self,
        idx,
        max_new_tokens,
        greedy=True
    ):
        if max_new_tokens == 0:
            return idx

        idx_cond = idx[
            :,
            -self.config.block_size:
        ]

        if idx_cond.shape[1] + max_new_tokens > self.config.block_size:
            past_key_values = None
            cache_len = 0

            for _ in range(max_new_tokens):
                if past_key_values is None or cache_len >= self.config.block_size:
                    idx_cond = idx[
                        :,
                        -self.config.block_size:
                    ]

                    past_key_values = self.init_kv_cache(
                        idx_cond.shape[0],
                        self.config.block_size,
                        idx_cond.device
                    )

                    logits, past_key_values, cache_len = self(
                        idx_cond,
                        past_key_values=past_key_values,
                        cache_len=0,
                        use_cache=True
                    )
                else:
                    logits, past_key_values, cache_len = self(
                        idx[
                            :,
                            -1:
                        ],
                        past_key_values=past_key_values,
                        cache_len=cache_len,
                        use_cache=True
                    )

                logits = logits[
                    :,
                    -1,
                    :
                ]

                if greedy:
                    idx_next = torch.argmax(
                        logits,
                        dim=-1,
                        keepdim=True
                    )
                else:
                    probs = F.softmax(
                        logits,
                        dim=-1
                    )
                    idx_next = torch.multinomial(
                        probs,
                        num_samples=1
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
            self.config.block_size,
            idx_cond.shape[1] + max_new_tokens
        )

        past_key_values = self.init_kv_cache(
            idx.shape[0],
            max_cache_len,
            idx.device
        )

        logits, past_key_values, cache_len = self(
            idx_cond,
            past_key_values=past_key_values,
            cache_len=0,
            use_cache=True
        )

        logits = logits[
            :,
            -1,
            :
        ]

        if greedy:
            idx_next = torch.argmax(
                logits,
                dim=-1,
                keepdim=True
            )
        else:
            probs = F.softmax(
                logits,
                dim=-1
            )
            idx_next = torch.multinomial(
                probs,
                num_samples=1
            )

        idx = torch.cat(
            [
                idx,
                idx_next
            ],
            dim=1
        )

        for _ in range(max_new_tokens - 1):
            logits, past_key_values, cache_len = self(
                idx_next,
                past_key_values=past_key_values,
                cache_len=cache_len,
                use_cache=True
            )

            logits = logits[
                :,
                -1,
                :
            ]

            if greedy:
                idx_next = torch.argmax(
                    logits,
                    dim=-1,
                    keepdim=True
                )
            else:
                probs = F.softmax(
                    logits,
                    dim=-1
                )
                idx_next = torch.multinomial(
                    probs,
                    num_samples=1
                )

            idx = torch.cat(
                [
                    idx,
                    idx_next
                ],
                dim=1
            )

        return idx


def config_from_hf(hf_config):
    return GPT2Config(
        vocab_size=hf_config.vocab_size,
        block_size=hf_config.n_positions,
        n_layer=hf_config.n_layer,
        n_head=hf_config.n_head,
        n_embd=hf_config.n_embd,
        dropout=0.0
    )
