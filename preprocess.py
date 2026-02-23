import pandas as pd
import re
from collections import Counter
import pickle


def extract_utterances(dialog_str):
    utterances = re.findall(r"'(.*?)'|\"(.*?)\"", dialog_str)
    cleaned = []
    for u1, u2 in utterances:
        text = u1 if u1 else u2
        cleaned.append(text.strip().lower())
    return cleaned


def load_dialog_pairs(csv_path):
    df = pd.read_csv(csv_path)
    pairs = []

    for dialog_str in df["dialog"]:
        utterances = extract_utterances(dialog_str)

        for i in range(len(utterances) - 1):
            input_text = utterances[i]
            target_text = utterances[i + 1]

            if input_text and target_text:
                pairs.append((input_text, target_text))

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
    pairs = load_dialog_pairs("data/train.csv")
    print("Total pairs:", len(pairs))

    word2idx, idx2word = build_vocab(pairs)
    inputs, targets = prepare_data(pairs, word2idx)

    print("Sample input:", inputs[0])
    print("Sample target:", targets[0])

    with open("vocab.pkl", "wb") as f:
        pickle.dump((word2idx, idx2word), f)