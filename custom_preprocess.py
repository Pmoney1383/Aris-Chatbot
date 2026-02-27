import os
from collections import Counter
import pickle

DATA_FILE = "data/greeting_dataset.txt"


def load_dialog_pairs(_=None):
    """
    Load custom greeting dataset formatted as:
    [category] input => response
    """
    pairs = []

    if not os.path.exists(DATA_FILE):
        raise FileNotFoundError(f"{DATA_FILE} not found.")

    with open(DATA_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if not line:
                continue

            # Remove [category]
            if line.startswith("["):
                category = line.split("]", 1)[0][1:]
                content = line.split("]", 1)[1].strip()
            else:
                continue

            if "=>" not in content:
                continue

            inp, tgt = content.split("=>")
            inp = inp.strip().lower()
            tgt = tgt.strip().lower()

            # prepend category token
            inp = f"<{category}> " + inp

            pairs.append((inp, tgt))

            if "=>" not in line:
                continue

            inp, tgt = line.split("=>")
            inp = inp.strip().lower()
            tgt = tgt.strip().lower()

            if inp and tgt:
                pairs.append((inp, tgt))

    return pairs


def build_vocab(pairs, vocab_size=15000):
    counter = Counter()

    for inp, tgt in pairs:
        counter.update(inp.split())
        counter.update(tgt.split())

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

    return word2idx, idx2word


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


if __name__ == "__main__":
    pairs = load_dialog_pairs()
    print("Total pairs:", len(pairs))

    word2idx, idx2word = build_vocab(pairs)
    inputs, targets = prepare_data(pairs, word2idx)

    print("Sample input:", inputs[0])
    print("Sample target:", targets[0])

    with open("vocab.pkl", "wb") as f:
        pickle.dump((word2idx, idx2word), f)