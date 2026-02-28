import os
import re
from collections import Counter
import pickle
import numpy as np

DATA_FILE = "clean_tagged.txt"


# =========================================================
# LOAD DIALOGUE PAIRS FROM CLEAN TAGGED FILE
# =========================================================

def load_dialog_stream():
    """
    Loads dialogue as a continuous token stream:
    <me> message <eot>
    <other> message <eot>
    Double texting by same speaker is merged.
    """

    if not os.path.exists(DATA_FILE):
        raise FileNotFoundError(f"{DATA_FILE} not found.")

    messages = []

    with open(DATA_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            if line.startswith("<me>"):
                speaker = "<me>"
                msg = line.replace("<me>", "", 1).strip()
            elif line.startswith("<other>"):
                speaker = "<other>"
                msg = line.replace("<other>", "", 1).strip()
            else:
                continue

            msg = re.sub(r"\s+", " ", msg).strip().lower()

            if msg:
                messages.append((speaker, msg))

    # Merge consecutive same-speaker messages
    merged = []
    for speaker, msg in messages:
        if merged and merged[-1][0] == speaker:
            merged[-1] = (speaker, merged[-1][1] + " " + msg)
        else:
            merged.append((speaker, msg))

    # Build stream with <eot>
    stream = []
    for speaker, msg in merged:
        stream.append(speaker)
        stream.extend(msg.split())
        stream.append("<eot>")

    return stream


# =========================================================
# BUILD VOCAB
# =========================================================

def build_vocab_from_stream(stream, vocab_size=19000):
    counter = Counter(stream)

    print("Total unique tokens before limit:", len(counter))

    word2idx = {
        "<pad>": 0,
        "<unk>": 1,
        "<eot>": 2,
        "<me>": 3,
        "<other>": 4
    }

    most_common = counter.most_common(vocab_size - len(word2idx))

    for word, _ in most_common:
        if word not in word2idx:
            word2idx[word] = len(word2idx)

    idx2word = {idx: word for word, idx in word2idx.items()}

    print("Final vocabulary size:", len(word2idx))

    return word2idx, idx2word


# =========================================================
# MAIN
# =========================================================

if __name__ == "__main__":

    stream = load_dialog_stream()

    print("Total tokens in stream:", len(stream))
    print("First 40 tokens:", stream[:40])

    word2idx, idx2word = build_vocab_from_stream(stream)

    # Encode full stream
    encoded_stream = [
        word2idx.get(token, word2idx["<unk>"])
        for token in stream
    ]

    print("First 40 encoded tokens:", encoded_stream[:40])

    with open("vocab.pkl", "wb") as f:
        pickle.dump((word2idx, idx2word), f)

    with open("encoded_stream.pkl", "wb") as f:
        pickle.dump(encoded_stream, f)

    print("Preprocessing complete.")