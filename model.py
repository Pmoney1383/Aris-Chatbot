import torch
import torch.nn as nn

class Attention(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.attn = nn.Linear(hidden_size * 2, hidden_size)
        self.v = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, hidden, encoder_outputs):
        # hidden: [num_layers, batch, hidden]
        # encoder_outputs: [batch, seq_len, hidden]

        batch_size = encoder_outputs.size(0)
        seq_len = encoder_outputs.size(1)

        # Use last layer hidden
        hidden = hidden[-1]  # [batch, hidden]
        hidden = hidden.unsqueeze(1).repeat(1, seq_len, 1)

        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim=2)))
        attention = self.v(energy).squeeze(2)

        return torch.softmax(attention, dim=1)

class Encoder(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx=0)
        self.dropout = nn.Dropout(dropout)
        self.lstm = nn.LSTM(
            embed_size,
            hidden_size,
            batch_first=True,
            dropout=dropout,
            num_layers=2
        )

    def forward(self, x):
        embedded = self.dropout(self.embedding(x))
        outputs, (hidden, cell) = self.lstm(embedded)
        return outputs, hidden, cell


class Decoder(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx=0)
        self.dropout = nn.Dropout(dropout)

        self.lstm = nn.LSTM(
            embed_size + hidden_size,
            hidden_size,
            batch_first=True,
            dropout=dropout,
            num_layers=2
        )

        self.attention = Attention(hidden_size)

        self.fc = nn.Linear(hidden_size * 2, vocab_size)

    def forward(self, x, hidden, cell, encoder_outputs):
        # x: [batch, 1]
        embedded = self.dropout(self.embedding(x))

        attn_weights = self.attention(hidden, encoder_outputs)
        context = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs)

        lstm_input = torch.cat((embedded, context), dim=2)

        outputs, (hidden, cell) = self.lstm(lstm_input, (hidden, cell))

        output = outputs.squeeze(1)
        context = context.squeeze(1)

        prediction = self.fc(torch.cat((output, context), dim=1))

        return prediction.unsqueeze(1), hidden, cell

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, src, trg):
        encoder_outputs, hidden, cell = self.encoder(src)

        batch_size = trg.size(0)
        trg_len = trg.size(1)
        vocab_size = self.decoder.fc.out_features

        outputs = torch.zeros(batch_size, trg_len, vocab_size).to(src.device)

        input_token = trg[:, 0].unsqueeze(1)  # <start>

        for t in range(trg_len):
            output, hidden, cell = self.decoder(
                input_token, hidden, cell, encoder_outputs
            )

            outputs[:, t, :] = output.squeeze(1)

            if t + 1 < trg_len:
                input_token = trg[:, t + 1].unsqueeze(1)

        return outputs