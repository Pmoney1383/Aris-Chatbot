import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pickle

from preprocess import load_dialog_pairs, build_vocab, prepare_data
from model import Encoder, Decoder, Seq2Seq


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_LEN = 20
BATCH_SIZE = 64
EMBED_SIZE = 256
HIDDEN_SIZE = 256
EPOCHS = 20


class ChatDataset(Dataset):
    def __init__(self, inputs, targets):
        self.inputs = torch.tensor(inputs, dtype=torch.long)
        self.targets = torch.tensor(targets, dtype=torch.long)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]


# Load data
pairs = load_dialog_pairs("data/train.csv")
word2idx, idx2word = build_vocab(pairs, vocab_size=10000)
inputs, targets = prepare_data(pairs, word2idx, MAX_LEN)

dataset = ChatDataset(inputs, targets)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# Model
vocab_size = len(word2idx)
encoder = Encoder(vocab_size, EMBED_SIZE, HIDDEN_SIZE)
decoder = Decoder(vocab_size, EMBED_SIZE, HIDDEN_SIZE)
model = Seq2Seq(encoder, decoder).to(DEVICE)

criterion = nn.CrossEntropyLoss(ignore_index=0)
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0

    for src, trg in loader:
        src = src.to(DEVICE)
        trg = trg.to(DEVICE)

        decoder_input = trg[:, :-1]
        decoder_target = trg[:, 1:]

        outputs = model(src, decoder_input)

        outputs = outputs.reshape(-1, vocab_size)
        decoder_target = decoder_target.reshape(-1)

        loss = criterion(outputs, decoder_target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {total_loss/len(loader):.4f}")

# Save
torch.save(model.state_dict(), "chatbot_model.pt")

with open("vocab.pkl", "wb") as f:
    pickle.dump((word2idx, idx2word), f)