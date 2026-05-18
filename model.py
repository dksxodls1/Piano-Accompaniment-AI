# src/model.py
# ============================================================
#  Vocal → Piano Seq2Seq Transformer 모델 정의
# ============================================================

import math
import torch
import torch.nn as nn

import src.config as cfg
from src.vocab import PAD_IDX, VOCAB_SIZE


# ── 1. Positional Encoding ───────────────────────────────────
class PositionalEncoding(nn.Module):
    """Sinusoidal Positional Encoding"""

    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe  = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(
            torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


# ── 2. Seq2Seq Transformer ───────────────────────────────────
class VocalToPianoTransformer(nn.Module):
    """
    보컬 화음 시퀀스 → 피아노 화음 시퀀스 변환 모델
    Encoder-Decoder 구조의 Transformer
    """

    def __init__(self,
                 vocab_size: int = VOCAB_SIZE,
                 d_model:    int = cfg.D_MODEL,
                 n_heads:    int = cfg.N_HEADS,
                 n_layers:   int = cfg.N_LAYERS,
                 ffn_dim:    int = cfg.FFN_DIM,
                 dropout:  float = cfg.DROPOUT,
                 pad_idx:    int = PAD_IDX):
        super().__init__()
        self.d_model   = d_model
        self.src_embed = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
        self.tgt_embed = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
        self.pos_enc   = PositionalEncoding(
            d_model, max_len=cfg.SEQ_LEN + 10, dropout=dropout
        )
        self.transformer = nn.Transformer(
            d_model            = d_model,
            nhead              = n_heads,
            num_encoder_layers = n_layers,
            num_decoder_layers = n_layers,
            dim_feedforward    = ffn_dim,
            dropout            = dropout,
            batch_first        = True,
        )
        self.out_proj = nn.Linear(d_model, vocab_size)
        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def make_causal_mask(self, size: int) -> torch.Tensor:
        return torch.triu(torch.ones(size, size), diagonal=1).bool()

    # ── 학습용 Forward ───────────────────────────────────────
    def forward(self, src: torch.Tensor, tgt_in: torch.Tensor) -> torch.Tensor:
        tgt_len = tgt_in.size(1)
        src_key_padding_mask = (src    == PAD_IDX)
        tgt_key_padding_mask = (tgt_in == PAD_IDX)
        tgt_mask = self.make_causal_mask(tgt_len).to(src.device)

        src_emb = self.pos_enc(self.src_embed(src)    * math.sqrt(self.d_model))
        tgt_emb = self.pos_enc(self.tgt_embed(tgt_in) * math.sqrt(self.d_model))

        out = self.transformer(
            src_emb, tgt_emb,
            tgt_mask                = tgt_mask,
            src_key_padding_mask    = src_key_padding_mask,
            tgt_key_padding_mask    = tgt_key_padding_mask,
            memory_key_padding_mask = src_key_padding_mask,
        )
        return self.out_proj(out)

    # ── 추론용: 인코더 ────────────────────────────────────────
    def encode(self, src: torch.Tensor):
        src_key_padding_mask = (src == PAD_IDX)
        src_emb = self.pos_enc(self.src_embed(src) * math.sqrt(self.d_model))
        memory  = self.transformer.encoder(
            src_emb, src_key_padding_mask=src_key_padding_mask
        )
        return memory, src_key_padding_mask

    # ── 추론용: 디코더 1스텝 ─────────────────────────────────
    def decode_step(self, tgt: torch.Tensor,
                    memory: torch.Tensor,
                    src_key_padding_mask: torch.Tensor) -> torch.Tensor:
        tgt_len  = tgt.size(1)
        tgt_mask = self.make_causal_mask(tgt_len).to(tgt.device)
        tgt_emb  = self.pos_enc(self.tgt_embed(tgt) * math.sqrt(self.d_model))
        out = self.transformer.decoder(
            tgt_emb, memory,
            tgt_mask                = tgt_mask,
            memory_key_padding_mask = src_key_padding_mask,
        )
        return self.out_proj(out[:, -1, :])
