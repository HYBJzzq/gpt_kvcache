import torch
import torch.nn as nn
import torch.nn.functional as F

from config import *

"""
Char-level GPT baseline.

This file intentionally keeps the original no-cache architecture used for
training and as the benchmark baseline. Experimental inference features
such as KV cache and static batching live in model_kvcache.py.
"""


class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()

        self.key = nn.Linear(
            n_embd,
            head_size,
            bias=False
        )

        self.query = nn.Linear(
            n_embd,
            head_size,
            bias=False
        )

        self.value = nn.Linear(
            n_embd,
            head_size,
            bias=False
        )

        self.head_size = head_size

        self.register_buffer(
            "tril",
            torch.tril(
                torch.ones(
                    block_size,
                    block_size
                )
            )
        )

        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape

        q = self.query(x)
        k = self.key(x)
        v = self.value(x)

        att = (
            q @ k.transpose(-2, -1)
        ) * (self.head_size ** -0.5)

        att = att.masked_fill(
            self.tril[:T, :T] == 0,
            float("-inf")
        )

        att = F.softmax(
            att,
            dim=-1
        )

        att = self.drop(att)

        return att @ v


class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        num_head,
        head_size
    ):
        super().__init__()

        self.heads = nn.ModuleList(
            [
                Head(head_size)
                for _ in range(num_head)
            ]
        )

        self.proj = nn.Linear(
            n_embd,
            n_embd
        )

        self.drop = nn.Dropout(
            dropout
        )

    def forward(self, x):
        x = torch.cat(
            [
                head(x)
                for head in self.heads
            ],
            dim=-1
        )

        x = self.proj(x)
        x = self.drop(x)

        return x


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

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    def __init__(self):
        super().__init__()

        head_size = n_embd // n_head

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

    def forward(self, x):
        x = self.ln1(x)
        x = x + self.sa(x)
        x = self.ln2(x)
        x = x + self.ffwd(x)

        return x


class GPT(nn.Module):
    def __init__(self, vocab_size):
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
                for _ in range(n_layer)
            ]
        )

        self.ln_f = nn.LayerNorm(
            n_embd
        )

        self.lm_head = nn.Linear(
            n_embd,
            vocab_size
        )

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
        targets=None
    ):
        B, T = idx.shape

        pos = torch.arange(
            T,
            device=idx.device
        )

        tok_emb = self.token_embedding_table(
            idx
        )

        pos_emb = self.position_embedding_table(
            pos
        )

        x = tok_emb + pos_emb

        for block in self.blocks:
            x = block(x)

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

            loss = F.cross_entropy(
                logits,
                targets
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
        for _ in range(max_new_tokens):
            idx_cond = idx[
                :,
                -block_size:
            ]

            logits, _ = self(idx_cond)

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
