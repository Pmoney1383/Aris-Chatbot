import math
import torch
import torch.nn as nn

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(1)
        self.register_buffer("pe", pe)

    def forward(self, x):
        # x shape: (seq_len, batch, d_model)
        seq_len = x.size(0)
        return x + self.pe[:seq_len]


class TransformerChatModel(nn.Module):
    def __init__(
        self,
        vocab_size,
        d_model,
        nhead,
        num_encoder_layers,
        num_decoder_layers,
        dim_feedforward,
        dropout,
        pad_idx,
    ):
        super().__init__()

        self.pad_idx = pad_idx
        self.d_model = d_model

        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
        self.pos_encoder = PositionalEncoding(d_model)

        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
        )

        self.fc_out = nn.Linear(d_model, vocab_size)

        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def make_src_padding_mask(self, src):
        # src: (batch, seq_len)
        return (src == self.pad_idx)

    def make_tgt_padding_mask(self, tgt):
        return (tgt == self.pad_idx)

    def forward(self, src, tgt):
        # src, tgt shape: (batch, seq_len)

        src_mask = None

        tgt_len = tgt.size(1)
        tgt_mask = torch.triu(
            torch.ones(tgt_len, tgt_len, device=tgt.device),
            diagonal=1
        ).bool()

        src_padding_mask = self.make_src_padding_mask(src)
        tgt_padding_mask = self.make_tgt_padding_mask(tgt)

        # Convert to (seq_len, batch)
        src = src.transpose(0, 1)
        tgt = tgt.transpose(0, 1)

        src_emb = self.embedding(src) * math.sqrt(self.d_model)
        tgt_emb = self.embedding(tgt) * math.sqrt(self.d_model)

        src_emb = self.pos_encoder(src_emb)
        tgt_emb = self.pos_encoder(tgt_emb)

        output = self.transformer(
            src_emb,
            tgt_emb,
            src_mask=src_mask,
            tgt_mask=tgt_mask,
            src_key_padding_mask=src_padding_mask,
            tgt_key_padding_mask=tgt_padding_mask,
            memory_key_padding_mask=src_padding_mask,  # IMPORTANT
        )
        output = self.fc_out(output)
        return output.transpose(0, 1)