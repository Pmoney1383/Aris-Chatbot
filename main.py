import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from rich.progress import Progress, BarColumn, TextColumn
from torchinfo import summary
import pickle
import time
import matplotlib.pyplot as plt
#from preprocess import load_dialog_pairs, build_vocab, prepare_data
#from custom_preprocess import load_dialog_pairs, build_vocab, prepare_data
from message_preprocess import load_dialog_stream, build_vocab_from_stream
from model import DecoderOnlyTransformer

from rich.progress import ProgressColumn
from rich.text import Text


# =========================================================
# PROGRESS COLUMN
# =========================================================

class SafeTimeRemainingColumn(ProgressColumn):
    def render(self, task):
        remaining = task.time_remaining
        if remaining is None:
            return Text("--s remaining")
        return Text(f"{int(remaining)}s remaining")


# =========================================================
# CONFIG
# =========================================================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", DEVICE)

MAX_LEN = 50
BATCH_SIZE = 64
EPOCHS = 15
VOCAB_LIMIT = 19000
LEARNING_RATE = 2e-4


# =========================================================
# LOAD DATA AND BUILD STREAM
# =========================================================

stream = load_dialog_stream()
print("Total tokens in stream:", len(stream))


# =========================================================
# BUILD VOCAB
# =========================================================

word2idx, idx2word = build_vocab_from_stream(stream, vocab_size=VOCAB_LIMIT)
PAD_IDX = word2idx["<pad>"]

encoded = [word2idx.get(token, word2idx["<unk>"]) for token in stream]

print("Final vocabulary size:", len(word2idx))


# =========================================================
# SLIDING WINDOW DATASET
# =========================================================

class StreamDataset(Dataset):
    def __init__(self, data, seq_len):
        self.data = data
        self.seq_len = seq_len

    def __len__(self):
        return len(self.data) - self.seq_len

    def __getitem__(self, idx):
        x = self.data[idx : idx + self.seq_len]
        y = self.data[idx + 1 : idx + self.seq_len + 1]
        return torch.tensor(x), torch.tensor(y)


# Build full dataset from entire token stream
full_dataset = StreamDataset(encoded, MAX_LEN)

# 90/10 random split at window level
train_size = int(0.9 * len(full_dataset))
val_size = len(full_dataset) - train_size

train_dataset, val_dataset = torch.utils.data.random_split(
    full_dataset,
    [train_size, val_size]
)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

print("Train sequences:", len(train_dataset))
print("Validation sequences:", len(val_dataset))




# =========================================================
# MODEL
# =========================================================

vocab_size = len(word2idx)

model = DecoderOnlyTransformer(
    vocab_size=vocab_size,
    d_model=512,
    nhead=16,
    num_layers=6,
    dim_feedforward=1024,
    dropout=0.3,
    pad_idx=PAD_IDX,
    max_len=MAX_LEN
).to(DEVICE)

print("\nModel Summary:\n")
summary(
    model,
    input_data=torch.zeros(1, MAX_LEN, dtype=torch.long).to(DEVICE),
)

criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX, label_smoothing=0.1)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)


# =========================================================
# TRAINING LOOP WITH PROGRESS BAR
# =========================================================

train_losses = []
train_accuracies = []

for epoch in range(EPOCHS):

    model.train()
    total_loss = 0
    correct = 0
    total_tokens = 0
    epoch_start_time = time.time()

    with Progress(
        TextColumn(f"Epoch {epoch+1}/{EPOCHS}"),
        BarColumn(bar_width=None, style="white", complete_style="green"),
        TextColumn("{task.completed}/{task.total}"),
        SafeTimeRemainingColumn(),
        TextColumn(" - acc: {task.fields[acc]:.4f}"),
        TextColumn(" - loss: {task.fields[loss]:.4f}"),
        transient=True,
    ) as progress:

        task = progress.add_task("", total=len(train_loader), acc=0.0, loss=0.0)

        for step, (x, y) in enumerate(train_loader):

            x = x.to(DEVICE)
            y = y.to(DEVICE)

            logits = model(x)

            logits_flat = logits.reshape(-1, vocab_size)
            y_flat = y.reshape(-1)

            loss = criterion(logits_flat, y_flat)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()

            preds = logits_flat.argmax(dim=1)
            mask = y_flat != PAD_IDX

            correct += (preds[mask] == y_flat[mask]).sum().item()
            total_tokens += mask.sum().item()

            current_acc = correct / total_tokens if total_tokens > 0 else 0

            progress.update(
                task,
                advance=1,
                acc=current_acc,
                loss=loss.item(),
            )

    avg_loss = total_loss / len(train_loader)
    accuracy = correct / total_tokens

    train_losses.append(avg_loss)
    train_accuracies.append(accuracy)

    torch.save(model.state_dict(), "chatbot_model.pt")

    print(f"\nEpoch {epoch+1}")
    print(f"Train Loss: {avg_loss:.4f} | Train Acc: {accuracy:.4f}")
    print("-" * 60)


# =========================================================
# ================= VALIDATION =================
# =========================================================


    model.eval()
    val_loss = 0
    val_correct = 0
    val_tokens_total = 0

    with torch.no_grad():
        for x, y in val_loader:
            x = x.to(DEVICE)
            y = y.to(DEVICE)

            logits = model(x)

            logits_flat = logits.reshape(-1, vocab_size)
            y_flat = y.reshape(-1)

            loss = criterion(logits_flat, y_flat)
            val_loss += loss.item()

            preds = logits_flat.argmax(dim=1)
            mask = y_flat != PAD_IDX

            val_correct += (preds[mask] == y_flat[mask]).sum().item()
            val_tokens_total += mask.sum().item()

    val_avg_loss = val_loss / len(val_loader)
    val_accuracy = val_correct / val_tokens_total

    print(f"Val Loss:   {val_avg_loss:.4f} | Val Acc: {val_accuracy:.4f}")
    print("-" * 60)

# =========================================================
# SAVE VOCAB
# =========================================================

with open("vocab.pkl", "wb") as f:
    pickle.dump((word2idx, idx2word), f)

print("\nTraining complete.")


# =========================================================
# PLOTS
# =========================================================

epochs_range = range(1, len(train_losses) + 1)

plt.figure()
plt.plot(epochs_range, train_losses)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss")
plt.savefig("loss_curve.png")
plt.show()

plt.figure()
plt.plot(epochs_range, train_accuracies)
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Training Accuracy")
plt.savefig("accuracy_curve.png")
plt.show()