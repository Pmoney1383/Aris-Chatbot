import tensorflow as tf

def build_model(vocab_size, embedding_dim=128, units=128, max_len=20):

    # Encoder
    encoder_inputs = tf.keras.Input(shape=(max_len,))
    embedding_layer = tf.keras.layers.Embedding(vocab_size, embedding_dim)

    encoder_embedding = embedding_layer(encoder_inputs)
    encoder_lstm = tf.keras.layers.LSTM(units, return_state=True)
    _, state_h, state_c = encoder_lstm(encoder_embedding)

    encoder_states = [state_h, state_c]

    # Decoder
    decoder_inputs = tf.keras.Input(shape=(max_len - 1,))
    decoder_embedding = embedding_layer(decoder_inputs)

    decoder_lstm = tf.keras.layers.LSTM(
        units,
        return_sequences=True,
        return_state=True
    )

    decoder_outputs, _, _ = decoder_lstm(
        decoder_embedding,
        initial_state=encoder_states
    )

    dense = tf.keras.layers.Dense(vocab_size, activation="softmax")
    outputs = dense(decoder_outputs)

    model = tf.keras.Model(
        [encoder_inputs, decoder_inputs],
        outputs
    )

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model