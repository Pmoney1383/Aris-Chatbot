import torch
import torch.nn.functional as F
import pickle
from model import Encoder, Decoder, Seq2Seq

MAX_LEN = 20
EMBED_SIZE = 256
HIDDEN_SIZE = 256

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load vocab
with open("vocab.pkl", "rb") as f:
    word2idx, idx2word = pickle.load(f)

vocab_size = len(word2idx)

# Load model
encoder = Encoder(vocab_size, EMBED_SIZE, HIDDEN_SIZE)
decoder = Decoder(vocab_size, EMBED_SIZE, HIDDEN_SIZE)
model = Seq2Seq(encoder, decoder)
model.load_state_dict(torch.load("chatbot_model.pt", map_location=DEVICE))
model.to(DEVICE)
model.eval()


def encode_input(text):
    tokens = text.lower().split()
    ids = [word2idx.get(w, word2idx["<unk>"]) for w in tokens]
    ids = ids[:MAX_LEN]
    ids += [word2idx["<pad>"]] * (MAX_LEN - len(ids))
    return torch.tensor([ids], dtype=torch.long).to(DEVICE)


def generate_response(text, temperature=0.8):
    src = encode_input(text)
    hidden, cell = model.encoder(src)

    input_token = torch.tensor([[word2idx["<start>"]]], dtype=torch.long).to(DEVICE)

    response = []

    for _ in range(MAX_LEN):
        output, hidden, cell = model.decoder(input_token, hidden, cell)
        logits = output[:, -1, :]
        probs = F.softmax(logits / temperature, dim=-1)

        predicted = torch.multinomial(probs, 1).item()

        if predicted == word2idx["<end>"]:
            break

        response.append(idx2word.get(predicted, "<unk>"))
        input_token = torch.tensor([[predicted]], dtype=torch.long).to(DEVICE)

    return " ".join(response)


if __name__ == "__main__":
    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            break

        reply = generate_response(user_input)
        print("Aris:", reply)