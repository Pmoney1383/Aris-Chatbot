import torch
import torch.nn.functional as F
import pickle

from model import TransformerChatModel


MAX_LEN = 30
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Load vocab
with open("vocab.pkl", "rb") as f:
    word2idx, idx2word = pickle.load(f)

vocab_size = len(word2idx)


# Load model (must match main.py hyperparams)
model = TransformerChatModel(
    vocab_size=vocab_size,
    d_model = 256,
    nhead = 8,
    num_encoder_layers = 4,
    num_decoder_layers = 4,
    dim_feedforward = 1024,
    dropout=0.1,
    pad_idx=word2idx["<pad>"],
).to(DEVICE)

model.load_state_dict(torch.load("chatbot_model.pt", map_location=DEVICE))
model.eval()


def encode_input(text: str) -> torch.Tensor:
    tokens = text.lower().split()
    ids = [word2idx.get(w, word2idx["<unk>"]) for w in tokens]
    ids = ids[:MAX_LEN]
    ids += [word2idx["<pad>"]] * (MAX_LEN - len(ids))
    return torch.tensor([ids], dtype=torch.long, device=DEVICE)


@torch.no_grad()
def generate_response(
    text: str,
    temperature: float = 0.7,
    top_k: int = 8,
    min_len: int = 3,
    max_len: int = MAX_LEN,
) -> str:

    src = encode_input(text)

    # Start with <start>
    generated = torch.tensor([[word2idx["<start>"]]], dtype=torch.long, device=DEVICE)

    for step in range(max_len):

        # model returns (batch, tgt_len, vocab)
        logits = model(src, generated)
        next_logits = logits[:, -1, :]  # last token position

        probs = F.softmax(next_logits / temperature, dim=-1)

        # top-k sampling
        top_probs, top_idx = torch.topk(probs, k=top_k, dim=-1)
        sampled = torch.multinomial(top_probs, num_samples=1)  # index into top_probs
        next_token = top_idx.gather(1, sampled).item()


        # append token (even if it's <end> early, we continue until min_len)
        generated = torch.cat(
            [generated, torch.tensor([[next_token]], dtype=torch.long, device=DEVICE)],
            dim=1
        )

    # Convert tokens to words, skipping <start> and stopping at <end>
    out_tokens = []
    for token_id in generated[0].tolist()[1:]:
        if token_id == word2idx["<end>"]:
            break
        out_tokens.append(idx2word.get(token_id, "<unk>"))

    return " ".join(out_tokens)


if __name__ == "__main__":
    while True:
        user_input = input("You: ").strip()
        if user_input.lower() == "exit":
            break
        reply = generate_response(user_input)
        print("Aris:", reply)