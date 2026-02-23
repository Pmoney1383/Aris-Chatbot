import pandas as pd
import re

def extract_utterances(dialog_str):
    # Extract text inside single or double quotes
    utterances = re.findall(r"'(.*?)'|\"(.*?)\"", dialog_str)

    # re.findall returns tuples because of two groups
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


if __name__ == "__main__":
    pairs = load_dialog_pairs("data/train.csv")

    print("Total pairs:", len(pairs))
    print("Sample pair:", pairs[0])