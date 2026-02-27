import os
import re
from collections import Counter
import pickle
import numpy as np

DATA_FILE = "clean_tagged.txt"


# =========================================================
# LOAD DIALOGUE PAIRS FROM CLEAN TAGGED FILE
# =========================================================

def load_dialog_pairs(_=None):
    """
    Loads dialogue pairs from clean_tagged.txt formatted as:
    <me> message
    <other> message
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
                speaker = "me"
                msg = line.replace("<me>", "", 1).strip()
            elif line.startswith("<other>"):
                speaker = "other"
                msg = line.replace("<other>", "", 1).strip()
            else:
                continue

            # normalize spacing
            msg = re.sub(r"\s+", " ", msg)
            msg = msg.strip()

            if msg:
                messages.append((speaker, msg))

    # =====================================================
    # CREATE DIALOGUE PAIRS (me -> other or other -> me)
    # =====================================================

    pairs = []

    for i in range(len(messages) - 1):
        speaker1, msg1 = messages[i]
        speaker2, msg2 = messages[i + 1]

        # Only create pair if speakers alternate
        if speaker1 != speaker2:
            pairs.append((msg1.lower(), msg2.lower()))

    return pairs


# =========================================================
# BUILD VOCAB
# =========================================================

def build_vocab(pairs, vocab_size=19000):
    counter = Counter()

    for inp, tgt in pairs:
        counter.update(inp.split())
        counter.update(tgt.split())

    print("Total unique tokens before limit:", len(counter))
    most_common = counter.most_common(vocab_size - 4)

    word2idx = {
        "<pad>": 0,
        "<unk>": 1,
        "<start>": 2,
        "<end>": 3
    }

    for word, _ in most_common:
        word2idx[word] = len(word2idx)

    idx2word = {idx: word for word, idx in word2idx.items()}
    print("Final vocabulary size (after limit):", len(word2idx))

    return word2idx, idx2word


# =========================================================
# ENCODING
# =========================================================

def encode_sentence(sentence, word2idx, max_len):
    tokens = sentence.split()
    ids = [word2idx.get(word, word2idx["<unk>"]) for word in tokens]

    ids = ids[:max_len]
    ids += [word2idx["<pad>"]] * (max_len - len(ids))

    return ids


def prepare_data(pairs, word2idx, max_len=20):
    inputs = []
    targets = []

    for inp, tgt in pairs:
        tgt = "<start> " + tgt + " <end>"

        input_ids = encode_sentence(inp, word2idx, max_len)
        target_ids = encode_sentence(tgt, word2idx, max_len)

        inputs.append(input_ids)
        targets.append(target_ids)

    return inputs, targets


# =========================================================
# MAIN
# =========================================================

if __name__ == "__main__":
    pairs = load_dialog_pairs()
    print("Total dialogue pairs:", len(pairs))

    # ================= LENGTH ANALYSIS =================

    input_lengths = []
    target_lengths = []

    for inp, tgt in pairs:
        input_lengths.append(len(inp.split()))
        target_lengths.append(len(tgt.split()))

    avg_input = sum(input_lengths) / len(input_lengths)
    avg_target = sum(target_lengths) / len(target_lengths)

    max_input = max(input_lengths)
    max_target = max(target_lengths)

    print("\n===== LENGTH STATS =====")
    print(f"Average input length:  {avg_input:.2f} tokens")
    print(f"Average target length: {avg_target:.2f} tokens")
    print(f"Max input length:      {max_input}")
    print(f"Max target length:     {max_target}")
    

    p95_input = np.percentile(input_lengths, 95)
    p95_target = np.percentile(target_lengths, 95)

    print(f"95th percentile input length:  {p95_input:.0f}")
    print(f"95th percentile target length: {p95_target:.0f}")
    word2idx, idx2word = build_vocab(pairs)
    inputs, targets = prepare_data(pairs, word2idx)

    print("Sample input:", pairs[0])
    print("Encoded input:", inputs[0])
    print("Encoded target:", targets[0])

    with open("vocab.pkl", "wb") as f:
        pickle.dump((word2idx, idx2word), f)

    print("Vocab saved.")