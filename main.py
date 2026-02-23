import tensorflow as tf
import pickle
from preprocess import load_dialog_pairs, prepare_data
from model import build_model

MAX_LEN = 20
VOCAB_SIZE = 10000

pairs = load_dialog_pairs("data/train.csv")
input_padded, target_padded, tokenizer = prepare_data(
    pairs,
    vocab_size=VOCAB_SIZE,
    max_len=MAX_LEN
)

decoder_input = target_padded[:, :-1]
decoder_target = target_padded[:, 1:]
decoder_target = decoder_target[..., tf.newaxis]

model = build_model(VOCAB_SIZE, max_len=MAX_LEN)

model.summary()

history = model.fit(
    [input_padded, decoder_input],
    decoder_target,
    batch_size=64,
    epochs=20
)

model.save("chatbot_model.keras")

with open("tokenizer.pkl", "wb") as f:
    pickle.dump(tokenizer, f)