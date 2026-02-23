import tensorflow as tf
import numpy as np
import pickle

MAX_LEN = 20
VOCAB_SIZE = 5000

# Load model
model = tf.keras.models.load_model("chatbot_model.keras")

# Load tokenizer
with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

index_word = tokenizer.index_word
word_index = tokenizer.word_index


def generate_response(input_text):
    input_text = input_text.lower()

    # Convert input to sequence
    input_seq = tokenizer.texts_to_sequences([input_text])
    input_seq = tf.keras.preprocessing.sequence.pad_sequences(
        input_seq,
        maxlen=MAX_LEN,
        padding='post',
        truncating='post'
    )

    # Start token
    start_token = word_index.get("<start>")
    end_token = word_index.get("<end>")

    decoder_input = np.zeros((1, MAX_LEN - 1))
    decoder_input[0, 0] = start_token

    response = []

    for i in range(1, MAX_LEN - 1):
        prediction = model.predict([input_seq, decoder_input], verbose=0)

        k = 20
        probs = prediction[0, i - 1]

        temperature = 0.8
        probs = np.log(probs + 1e-9) / temperature
        probs = np.exp(probs)
        probs = probs / np.sum(probs)

        top_k_indices = np.argsort(probs)[-k:]
        top_k_probs = probs[top_k_indices]
        top_k_probs = top_k_probs / np.sum(top_k_probs)

        predicted_token = np.random.choice(top_k_indices, p=top_k_probs)

        if predicted_token == end_token:
            break

        word = index_word.get(predicted_token, "")
        response.append(word)

        decoder_input[0, i] = predicted_token

    return " ".join(response)


if __name__ == "__main__":
    while True:
        user_input = input("You: ")

        if user_input.lower() == "exit":
            break

        bot_response = generate_response(user_input)
        print("Aris:", bot_response)