import torch
import torch.nn as nn
import torch.nn.functional as F


# =========================
# Model Configuration
# =========================

class IPLGPTConfig:
    def __init__(
        self,
        vocab_size,
        block_size=1024,
        n_layer=8,
        n_head=8,
        n_embd=512,
        dropout=0.1,
        bias=True
    ):
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.n_layer = n_layer
        self.n_head = n_head
        self.n_embd = n_embd
        self.dropout = dropout
        self.bias = bias


# =========================
# Attention
# =========================

class MaskedMultiHeadSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()

        self.n_head = config.n_head
        self.n_embd = config.n_embd

        self.qkv_proj = nn.Linear(
            config.n_embd,
            3 * config.n_embd,
            bias=config.bias
        )

        self.out_proj = nn.Linear(
            config.n_embd,
            config.n_embd,
            bias=config.bias
        )

        self.register_buffer(
            "causal_mask",
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
            )
        )

    def forward(self, x):

        B, T, C = x.size()

        q, k, v = self.qkv_proj(x).split(
            self.n_embd,
            dim=2
        )

        head_dim = C // self.n_head

        q = q.view(
            B, T, self.n_head, head_dim
        ).transpose(1, 2)

        k = k.view(
            B, T, self.n_head, head_dim
        ).transpose(1, 2)

        v = v.view(
            B, T, self.n_head, head_dim
        ).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) * (
            1.0 / (head_dim ** 0.5)
        )

        att = att.masked_fill(
            self.causal_mask[:, :, :T, :T] == 0,
            float("-inf")
        )

        att = F.softmax(att, dim=-1)

        y = att @ v

        y = y.transpose(
            1,
            2
        ).contiguous().view(
            B,
            T,
            C
        )

        y = self.out_proj(y)

        return y


# =========================
# Feed Forward
# =========================

class FeedForward(nn.Module):

    def __init__(self, config):
        super().__init__()

        self.fc1 = nn.Linear(
            config.n_embd,
            4 * config.n_embd,
            bias=config.bias
        )

        self.fc2 = nn.Linear(
            4 * config.n_embd,
            config.n_embd,
            bias=config.bias
        )

        self.activation = nn.GELU()

    def forward(self, x):

        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)

        return x


# =========================
# Decoder Block
# =========================

class DecoderBlock(nn.Module):

    def __init__(self, config):
        super().__init__()

        self.ln1 = nn.LayerNorm(
            config.n_embd
        )

        self.attn = MaskedMultiHeadSelfAttention(
            config
        )

        self.ln2 = nn.LayerNorm(
            config.n_embd
        )

        self.ffn = FeedForward(
            config
        )

    def forward(self, x):

        x = x + self.attn(
            self.ln1(x)
        )

        x = x + self.ffn(
            self.ln2(x)
        )

        return x


# =========================
# GPT Model
# =========================

class IPLGPT(nn.Module):

    def __init__(self, config):
        super().__init__()

        self.config = config

        self.token_embedding = nn.Embedding(
            config.vocab_size,
            config.n_embd
        )

        self.position_embedding = nn.Embedding(
            config.block_size,
            config.n_embd
        )

        self.blocks = nn.ModuleList(
            [
                DecoderBlock(config)
                for _ in range(config.n_layer)
            ]
        )

        self.ln_final = nn.LayerNorm(
            config.n_embd
        )

        self.lm_head = nn.Linear(
            config.n_embd,
            config.vocab_size,
            bias=False
        )

    def forward(self, idx):

        B, T = idx.size()

        pos = torch.arange(
            0,
            T,
            dtype=torch.long,
            device=idx.device
        )

        tok_emb = self.token_embedding(idx)

        pos_emb = self.position_embedding(pos)

        x = tok_emb + pos_emb

        for block in self.blocks:
            x = block(x)

        x = self.ln_final(x)

        logits = self.lm_head(x)

        return logits