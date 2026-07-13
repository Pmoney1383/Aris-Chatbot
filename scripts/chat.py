"""Conversation REPL for the SFT'd Aris chatbot.

Loads the newest checkpoint from checkpoints/sft/ (falls back to checkpoints/
if SFT hasn't run yet). Keeps the running conversation as a flat token id
list, formats each user message with the chat template from src/chat_format.py,
and stops generation at the <|eot|> delimiter.

Commands: `clear` resets the conversation, `exit` / `quit` leave.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn.functional as F
from rich.console import Console

from src.chat_format import (
    EOT_IDS,
    NEWLINE_IDS,
    encode_chat_prompt,
    encode_conversation,
    get_encoding,
)
from src.config import GenerationConfig, ModelConfig, SFTConfig
from src.model import DecoderOnlyTransformer

console = Console()

HISTORY_LIMIT = 1024  # trim conversation history beyond this many tokens

# Invisible priming injected into the token history at the start of every
# session (and after `clear`) — never printed to the user.
PRIMING_TURNS = [
    ("user",
     'System: You are Aris, a casual and friendly AI assistant. You talk '
     'like a young person — use casual language, slang like "bro", "gang", '
     '"bet", "yurrr", "no cap". Keep responses short and punchy. Don\'t say '
     '"As an AI language model". Be helpful but chill about it. Who are you '
     'and how do you talk?'),
    ("assistant",
     "yooo i'm Aris, your AI homie 🔥 i keep it real and casual, no "
     "corporate speak fr. what you need gang?"),
    ("user", "okay bet, can you help me with something?"),
    ("assistant", "yurrr for sure bro what you need? i got you 🙌"),
]


def priming_ids() -> list[int]:
    ids, _ = encode_conversation(PRIMING_TURNS)
    return ids


def load_model(device):
    cfg = SFTConfig()
    ckpt_path = cfg.checkpoint_dir / "ckpt_003000.pt"
    if not ckpt_path.exists():
        console.print(f"[red]Checkpoint {ckpt_path} not found. Run "
                      "scripts/sft_train.py first.[/]")
        sys.exit(1)

    console.print(f"Loading [bold]{ckpt_path}[/]...")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    model_cfg = ModelConfig(**ckpt["model_config"])
    model = DecoderOnlyTransformer(model_cfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    console.print(f"Checkpoint step: {ckpt['step']}, loss: {ckpt['loss']:.4f}")
    return model


def trim_history(history: list[int]) -> list[int]:
    """Keep the most recent <= HISTORY_LIMIT tokens, cutting only at <|eot|>
    boundaries so a turn is never split mid-way."""
    if len(history) <= HISTORY_LIMIT:
        return history

    # candidate cut points: just after each <|eot|> (plus its newline)
    n = len(EOT_IDS)
    for i in range(len(history) - n + 1):
        if history[i : i + n] == EOT_IDS:
            end = i + n
            if history[end : end + len(NEWLINE_IDS)] == NEWLINE_IDS:
                end += len(NEWLINE_IDS)
            if len(history) - end <= HISTORY_LIMIT:
                return history[end:]
    # no boundary leaves us under the limit (single huge turn): hard trim
    return history[-HISTORY_LIMIT:]


@torch.no_grad()
def generate(model, context_ids, device, gen_cfg: GenerationConfig):
    """Generate until <|eot|> (returned WITHOUT the eot sequence)."""
    x = torch.tensor([context_ids], dtype=torch.long, device=device)
    generated = []

    for _ in range(gen_cfg.max_new_tokens):
        x_cond = x[:, -model.cfg.max_seq_len:]
        if device == "cuda":
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits, _ = model(x_cond)
        else:
            logits, _ = model(x_cond)
        logits = logits[:, -1, :].float()

        for tok in set(generated[-gen_cfg.repetition_window:]):
            logits[0, tok] /= gen_cfg.repetition_penalty

        logits /= gen_cfg.temperature
        v, _ = torch.topk(logits, gen_cfg.top_k)
        logits[logits < v[:, [-1]]] = -float("inf")
        probs = F.softmax(logits, dim=-1)
        next_tok = torch.multinomial(probs, 1)

        generated.append(next_tok.item())
        x = torch.cat([x, next_tok], dim=1)

        if generated[-len(EOT_IDS):] == EOT_IDS:
            return generated[: -len(EOT_IDS)]

    return generated


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    console.print(f"Device: {device}")

    model = load_model(device)
    enc = get_encoding()
    gen_cfg = GenerationConfig()

    console.print("\nChat with Aris. Commands: [bold]clear[/] resets the "
                  "conversation, [bold]exit[/]/[bold]quit[/] leave.\n")

    # flat token ids of the whole conversation, seeded with the hidden priming
    history: list[int] = priming_ids()

    while True:
        try:
            user_text = console.input("[bold cyan]you> [/]").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if user_text.lower() in ("exit", "quit"):
            break
        if user_text.lower() == "clear":
            history = priming_ids()
            console.print("[dim]conversation cleared[/]\n")
            continue
        if not user_text:
            continue

        history.extend(encode_chat_prompt(user_text))
        history = trim_history(history)

        reply_ids = generate(model, history, device, gen_cfg)
        reply = enc.decode(reply_ids).strip()

        # record the full assistant turn (with eot) so the template stays intact
        history.extend(reply_ids + EOT_IDS + NEWLINE_IDS)
        history = trim_history(history)

        console.print(f"[white]{reply}[/]\n")


if __name__ == "__main__":
    main()
