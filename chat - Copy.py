import torch
import torch.nn.functional as F
import pickle

from model import DecoderOnlyTransformer


# =========================================================
# CONFIG
# =========================================================

MAX_CONTEXT = 100        # how many past tokens model can see
MAX_GENERATE = 25        # max tokens to generate per reply
TEMPERATURE = 0.6
TOP_K = 3
REPETITION_PENALTY = 1.7

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =========================================================
# LOAD VOCAB
# =========================================================

with open("vocab.pkl", "rb") as f:
    word2idx, idx2word = pickle.load(f)

vocab_size = len(word2idx)


# =========================================================
# LOAD MODEL
# =========================================================

model = DecoderOnlyTransformer(
    vocab_size=vocab_size,
    d_model=256,
    nhead=8,
    num_layers=4,
    dim_feedforward=1024,
    dropout=0.1,
    pad_idx=word2idx["<pad>"],
    max_len=MAX_CONTEXT
).to(DEVICE)

model.load_state_dict(torch.load("chatbot_model.pt", map_location=DEVICE))
model.eval()


# =========================================================
# TOKEN UTILITIES
# =========================================================

def encode_text(text):
    tokens = text.lower().split()
    return [word2idx.get(w, word2idx["<unk>"]) for w in tokens]


def decode_tokens(token_ids):
    words = []
    for t in token_ids:
        words.append(idx2word.get(t, "<unk>"))
    return " ".join(words)


# =========================================================
# GENERATION
# =========================================================

def trim_to_context(history_tokens):
    if len(history_tokens) <= MAX_CONTEXT:
        return history_tokens

    trimmed = history_tokens[-MAX_CONTEXT:]

    # find first <eot> in trimmed window
    for i, token in enumerate(trimmed):
        if idx2word.get(token) == "<eot>":
            return trimmed[i+1:]

    # fallback if no <eot> found
    return trimmed

@torch.no_grad()
def generate_reply(history_tokens):

    input_ids = trim_to_context(history_tokens)
    generated = []

    for _ in range(MAX_GENERATE):

        input_tensor = torch.tensor([input_ids], dtype=torch.long, device=DEVICE)

        logits = model(input_tensor)
        next_logits = logits[:, -1, :]

        # repetition penalty
        for token in set(generated[-20:]):
            next_logits[0, token] /= REPETITION_PENALTY

        next_logits = next_logits / TEMPERATURE

        probs = F.softmax(next_logits, dim=-1)

        top_probs, top_indices = torch.topk(probs, TOP_K)
        sampled_index = torch.multinomial(top_probs, 1)
        next_token = top_indices.gather(1, sampled_index).item()

        # STOP if it tries to switch speaker
        if idx2word.get(next_token) == "<eot>":
            break

        generated.append(next_token)
        input_ids.append(next_token)

        if len(input_ids) > MAX_CONTEXT:
            input_ids = input_ids[-MAX_CONTEXT:]

    return generated

# =========================================================
# INTERACTIVE CHAT LOOP
# =========================================================

if __name__ == "__main__":

    conversation_history = []

    while True:
        user_input = input("You: ").strip()

        if user_input.lower() == "exit":
            break

        # Add user turn
        user_tokens = encode_text("<me> " + user_input + " <eot>")
        conversation_history.extend(user_tokens)

        # Generate model reply
        # Tell model it's its turn
        bot_prefix = encode_text("<other>")
        conversation_history.extend(bot_prefix)

        reply_tokens = generate_reply(conversation_history)
        conversation_history.extend(reply_tokens)
        conversation_history.append(word2idx["<eot>"])
        conversation_history = conversation_history[-MAX_CONTEXT:]
        # Add model turn
       # conversation_history.extend(reply_tokens)

        reply_text = decode_tokens(reply_tokens)

        print("Aris:", reply_text)