import pandas as pd
import re
import tensorflow as tf

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


def prepare_data(pairs, vocab_size=5000, max_len=20):
    input_texts = []
    target_texts = []

    for inp, tgt in pairs:
        input_texts.append(inp)
        target_texts.append("<start> " + tgt + " <end>")

    tokenizer = tf.keras.preprocessing.text.Tokenizer(
        num_words=vocab_size,
        filters='',
        oov_token="<unk>"
    )

    tokenizer.fit_on_texts(input_texts + target_texts)

    input_sequences = tokenizer.texts_to_sequences(input_texts)
    target_sequences = tokenizer.texts_to_sequences(target_texts)

    input_padded = tf.keras.preprocessing.sequence.pad_sequences(
        input_sequences,
        maxlen=max_len,
        padding='post',
        truncating='post'
    )

    target_padded = tf.keras.preprocessing.sequence.pad_sequences(
        target_sequences,
        maxlen=max_len,
        padding='post',
        truncating='post'
    )

    return input_padded, target_padded, tokenizer


if __name__ == "__main__":
    pairs = load_dialog_pairs("data/train.csv")
    print("Total pairs:", len(pairs))

    input_padded, target_padded, tokenizer = prepare_data(pairs)

    print("Input shape:", input_padded.shape)
    print("Target shape:", target_padded.shape)

    print("Sample input sequence:", input_padded[0])
    print("Sample target sequence:", target_padded[0])