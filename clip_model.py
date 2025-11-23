import torch
import torch.nn as nn
import numpy as np


class PositionalEmbedding(nn.Module):
    def __init__(self, width, max_seq_length):
        super().__init__()
        pe = torch.zeros(max_seq_length, width)
        for pos in range(max_seq_length):
            for i in range(width):
                if i % 2 == 0:
                    pe[pos, i] = np.sin(pos / (10000 ** (i / width)))
                else:
                    pe[pos, i] = np.cos(pos / (10000 ** ((i - 1) / width)))
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, : x.size(1), :]


class AttentionHead(nn.Module):
    def __init__(self, width, head_size):
        super().__init__()
        self.head_size = head_size
        self.query = nn.Linear(width, head_size)
        self.key = nn.Linear(width, head_size)
        self.value = nn.Linear(width, head_size)

    def forward(self, x, mask=None):
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)
        attention = Q @ K.transpose(-2, -1)
        attention = attention / (self.head_size ** 0.5)
        if mask is not None:
            attention = attention.masked_fill(mask == 0, float("-inf"))
        attention = torch.softmax(attention, dim=-1)
        return attention @ V


class MultiHeadAttention(nn.Module):
    def __init__(self, width, n_heads):
        super().__init__()
        self.head_size = width // n_heads
        self.heads = nn.ModuleList([AttentionHead(width, self.head_size) for _ in range(n_heads)])
        self.W_o = nn.Linear(width, width)

    def forward(self, x, mask=None):
        out = torch.cat([head(x, mask=mask) for head in self.heads], dim=-1)
        return self.W_o(out)


class TransformerEncoder(nn.Module):
    def __init__(self, width, n_heads, r_mlp=4, dropout=0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(width)
        self.mha = MultiHeadAttention(width, n_heads)
        self.dropout1 = nn.Dropout(dropout)
        self.ln2 = nn.LayerNorm(width)
        self.mlp = nn.Sequential(
            nn.Linear(width, width * r_mlp),
            nn.GELU(),
            nn.Linear(width * r_mlp, width),
        )
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        x = x + self.dropout1(self.mha(self.ln1(x), mask=mask))
        x = x + self.dropout2(self.mlp(self.ln2(x)))
        return x


class TextEncoder(nn.Module):
    def __init__(self, vocab_size, width, max_seq_length, n_heads, n_layers, emb_dim):
        super().__init__()
        self.max_seq_length = max_seq_length
        self.encoder_embedding = nn.Embedding(vocab_size, width)
        self.positional_embedding = PositionalEmbedding(width, max_seq_length)
        self.encoder = nn.ModuleList([TransformerEncoder(width, n_heads) for _ in range(n_layers)])
        self.projection = nn.Parameter(torch.randn(width, emb_dim))

    def forward(self, text, mask=None):
        x = self.encoder_embedding(text)
        x = self.positional_embedding(x)
        for layer in self.encoder:
            x = layer(x, mask=mask)
        if mask is not None:
            lengths = torch.sum(mask[:, 0], dim=1) - 1
            lengths = lengths.to(torch.long)
        else:
            lengths = torch.full((x.size(0),), x.size(1) - 1, device=x.device, dtype=torch.long)
        x = x[torch.arange(text.shape[0], device=text.device), lengths]
        x = x @ self.projection
        x = x / torch.norm(x, dim=-1, keepdim=True)
        return x


class AudioEncoder(nn.Module):
    def __init__(self, width, n_patches, emb_dim, n_layers, n_heads, patch_dim):
        super().__init__()
        self.max_seq_length = n_patches + 1
        self.linear_project = nn.Linear(patch_dim, width)
        self.cls_token = nn.Parameter(torch.randn(1, 1, width))
        self.positional_embedding = PositionalEmbedding(width, self.max_seq_length)
        self.encoder = nn.ModuleList([TransformerEncoder(width, n_heads) for _ in range(n_layers)])
        self.projection = nn.Parameter(torch.randn(width, emb_dim))

    def forward(self, patches):
        x = self.linear_project(patches)
        B = x.size(0)
        cls = self.cls_token.expand(B, 1, -1)
        x = torch.cat((cls, x), dim=1)
        x = self.positional_embedding(x)
        for layer in self.encoder:
            x = layer(x)
        x = x[:, 0]
        x = x @ self.projection
        x = x / torch.norm(x, dim=-1, keepdim=True)
        return x


class CLIP(nn.Module):
    def __init__(
        self,
        emb_dim=512,
        vit_width=512,
        n_patches=44,
        patch_dim=256,
        vit_layers=8,
        vit_heads=8,
        vocab_size=256,
        text_width=512,
        max_seq_length=32,
        text_heads=8,
        text_layers=8,
    ):
        super().__init__()
        self.audio_encoder = AudioEncoder(
            vit_width,
            n_patches,
            emb_dim,
            vit_layers,
            vit_heads,
            patch_dim,
        )
        self.text_encoder = TextEncoder(
            vocab_size,
            text_width,
            max_seq_length,
            text_heads,
            text_layers,
            emb_dim,
        )
        self.temperature = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def forward(self, audio_patches, text, mask=None):
        I_e = self.audio_encoder(audio_patches)
        T_e = self.text_encoder(text, mask=mask)
        logits = (I_e @ T_e.transpose(-2, -1)) * torch.exp(self.temperature)
        labels = torch.arange(logits.shape[0], device=logits.device)
        loss_i = nn.functional.cross_entropy(logits.transpose(-2, -1), labels)
        loss_t = nn.functional.cross_entropy(logits, labels)
        return (loss_i + loss_t) / 2
