import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from rich.progress import Progress, BarColumn, TextColumn
from torchinfo import summary
import pickle
import time
import matplotlib.pyplot as plt
from preprocess import load_dialog_pairs, build_vocab, prepare_data
from model import TransformerChatModel

from rich.progress import ProgressColumn
from rich.text import Text


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

MAX_LEN = 20
BATCH_SIZE = 64
EPOCHS = 20
VOCAB_LIMIT = 15000
EARLY_STOP_PATIENCE = 4


# =========================================================
# DATASET
# =========================================================

class ChatDataset(Dataset):
    def __init__(self, inputs, targets):
        self.inputs = torch.tensor(inputs, dtype=torch.long)
        self.targets = torch.tensor(targets, dtype=torch.long)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]


# =========================================================
# LOAD DATA
# =========================================================

pairs = load_dialog_pairs("data/train.csv")
print("Total dialog pairs:", len(pairs))

word2idx, idx2word = build_vocab(pairs, vocab_size=VOCAB_LIMIT)
inputs, targets = prepare_data(pairs, word2idx, MAX_LEN)

PAD_IDX = word2idx["<pad>"]

train_inputs, val_inputs, train_targets, val_targets = train_test_split(
    inputs, targets, test_size=0.1, random_state=42
)

train_dataset = ChatDataset(train_inputs, train_targets)
val_dataset = ChatDataset(val_inputs, val_targets)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)


# =========================================================
# MODEL
# =========================================================

vocab_size = len(word2idx)

model = TransformerChatModel(
    vocab_size=vocab_size,
    d_model=256,
    nhead=8,
    num_encoder_layers=4,
    num_decoder_layers=4,
    dim_feedforward=1024,
    dropout=0.3,
    pad_idx=PAD_IDX,
).to(DEVICE)

print("\nModel Summary:\n")
summary(
    model,
    input_data=(
        torch.zeros(1, MAX_LEN, dtype=torch.long).to(DEVICE),
        torch.zeros(1, MAX_LEN - 1, dtype=torch.long).to(DEVICE),
    ),
)

criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX, label_smoothing=0.1)

optimizer = torch.optim.Adam(model.parameters(), lr=3e-4, weight_decay=1e-4)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="min",
    factor=0.5,
    patience=2,
)


# =========================================================
# TRAINING LOOP
# =========================================================

best_val_loss = float("inf")
early_stop_counter = 0
train_losses = []
val_losses = []
train_accuracies = []
val_accuracies = []
for epoch in range(EPOCHS):

    model.train()
    train_loss = 0
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


        for step, (src, trg) in enumerate(train_loader):

            src = src.to(DEVICE)
            trg = trg.to(DEVICE)

            decoder_input = trg[:, :-1]
            decoder_target = trg[:, 1:]

            outputs = model(src, decoder_input)

            outputs_flat = outputs.reshape(-1, vocab_size)
            target_flat = decoder_target.reshape(-1)

            loss = criterion(outputs_flat, target_flat)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss += loss.item()

            preds = outputs_flat.argmax(dim=1)
            mask = target_flat != PAD_IDX

            correct += (preds[mask] == target_flat[mask]).sum().item()
            total_tokens += mask.sum().item()

            current_acc = correct / total_tokens if total_tokens > 0 else 0
            elapsed = time.time() - epoch_start_time
            steps_done = step + 1
            steps_left = len(train_loader) - steps_done
            avg_step_time = elapsed / steps_done
            remaining_seconds = int(avg_step_time * steps_left)
            ms_per_step = int(avg_step_time * 1000)
            eta_string = f"{remaining_seconds}s {ms_per_step}ms/step"
            progress.update(
                task,
                advance=1,
                acc=current_acc,
                loss=loss.item(),
            )

    train_loss /= len(train_loader)
    train_accuracy = correct / total_tokens


    # ================= VALIDATION =================

    model.eval()
    val_loss = 0
    val_correct = 0
    val_total_tokens = 0

    with torch.no_grad():
        for src, trg in val_loader:

            src = src.to(DEVICE)
            trg = trg.to(DEVICE)

            decoder_input = trg[:, :-1]
            decoder_target = trg[:, 1:]

            outputs = model(src, decoder_input)

            outputs_flat = outputs.reshape(-1, vocab_size)
            target_flat = decoder_target.reshape(-1)

            loss = criterion(outputs_flat, target_flat)
            val_loss += loss.item()

            preds = outputs_flat.argmax(dim=1)
            mask = target_flat != PAD_IDX

            val_correct += (preds[mask] == target_flat[mask]).sum().item()
            val_total_tokens += mask.sum().item()

    val_loss /= len(val_loader)
    val_accuracy = val_correct / val_total_tokens

    scheduler.step(float(val_loss))

    epoch_time = time.time() - epoch_start_time

    train_losses.append(train_loss)
    val_losses.append(val_loss)
    train_accuracies.append(train_accuracy)
    val_accuracies.append(val_accuracy)

    print(f"\nEpoch {epoch+1} Summary")
    print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_accuracy:.4f}")
    print(f"Val   Loss: {val_loss:.4f} | Val   Acc: {val_accuracy:.4f}")
    print(f"Epoch Time: {epoch_time:.2f}s")
    print("-" * 60)


    # ================= EARLY STOPPING =================

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        early_stop_counter = 0
        torch.save(model.state_dict(), "chatbot_model.pt")
        print("Best model saved.")
    else:
        early_stop_counter += 1
        if early_stop_counter >= EARLY_STOP_PATIENCE:
            print("Early stopping triggered.")
            break


# =========================================================
# SAVE VOCAB
# =========================================================

with open("vocab.pkl", "wb") as f:
    pickle.dump((word2idx, idx2word), f)

print("\nTraining complete.")


# =========================================================
# Analysis Plot
# =========================================================

epochs_range = range(1, EPOCHS + 1)

plt.figure()
plt.plot(epochs_range, train_losses)
plt.plot(epochs_range, val_losses)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training vs Validation Loss")
plt.legend(["Train Loss", "Validation Loss"])
plt.savefig("loss_curve.png")
plt.show()
plt.figure()
plt.plot(epochs_range, train_accuracies)
plt.plot(epochs_range, val_accuracies)
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Training vs Validation Accuracy")
plt.legend(["Train Accuracy", "Validation Accuracy"])
plt.savefig("accuracy_curve.png")
plt.show()
