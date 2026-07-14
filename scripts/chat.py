"""Conversation REPL for the SFT'd Aris chatbot.

Loads the newest checkpoint from checkpoints/sft_persona_v2/ if present,
otherwise checkpoints/sft/ckpt_003000.pt. Keeps the running conversation as a
flat token id list, formats each user message with the chat template from
src/chat_format.py, and stops generation at the <|eot|> delimiter.

Commands:
    clear            reset the conversation
    good             save the last exchange to data/raw/persona_curated.jsonl
    fix <text>       save the last user message with <text> as the corrected
                     assistant reply to persona_curated.jsonl
    exit / quit      leave
"""

import json
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
from src.config import GenerationConfig, ModelConfig, PersonaV2Config, SFTConfig
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
    persona_cfg = PersonaV2Config()
    persona_ckpts = sorted(persona_cfg.checkpoint_dir.glob("ckpt_*.pt"))
    if persona_ckpts:
        ckpt_path = persona_ckpts[-1]
        console.print(f"[bold green]Loading persona v2 checkpoint: {ckpt_path}[/]")
    else:
        ckpt_path = SFTConfig().checkpoint_dir / "ckpt_003000.pt"
        if not ckpt_path.exists():
            console.print(f"[red]No persona v2 checkpoint and {ckpt_path} not "
                          "found. Run scripts/sft_train.py first.[/]")
            sys.exit(1)
        console.print(f"[bold]Loading chat-SFT checkpoint: {ckpt_path}[/] "
                      "(no persona v2 checkpoint yet)")
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


def save_curated_pair(user_text: str, assistant_text: str) -> Path:
    """Append one {"user", "assistant"} pair to data/raw/persona_curated.jsonl."""
    path = PersonaV2Config().curated_file
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps({"user": user_text, "assistant": assistant_text},
                           ensure_ascii=False) + "\n")
    return path


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    console.print(f"Device: {device}")

    model = load_model(device)
    enc = get_encoding()
    gen_cfg = GenerationConfig()

    console.print("\nChat with Aris. Commands: [bold]clear[/] resets the "
                  "conversation, [bold]good[/] saves the last exchange, "
                  "[bold]fix <text>[/] saves it with a corrected reply, "
                  "[bold]exit[/]/[bold]quit[/] leave.\n")

    # flat token ids of the whole conversation, seeded with the hidden priming
    history: list[int] = priming_ids()
    last_user: str | None = None
    last_reply: str | None = None

    while True:
        try:
            user_text = console.input("[bold cyan]you> [/]").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if user_text.lower() in ("exit", "quit"):
            break
        if user_text.lower() == "clear":
            history = priming_ids()
            last_user = last_reply = None
            console.print("[dim]conversation cleared[/]\n")
            continue
        if user_text.lower() == "good":
            if last_user is None or last_reply is None:
                console.print("[yellow]nothing to save yet[/]\n")
            else:
                path = save_curated_pair(last_user, last_reply)
                console.print(f"[green]saved exchange to {path.name}[/]\n")
            continue
        if user_text.lower().startswith("fix ") or user_text.lower() == "fix":
            corrected = user_text[3:].strip()
            if last_user is None:
                console.print("[yellow]nothing to fix yet[/]\n")
            elif not corrected:
                console.print("[yellow]usage: fix <corrected assistant reply>[/]\n")
            else:
                path = save_curated_pair(last_user, corrected)
                console.print(f"[green]saved corrected exchange to {path.name}[/]\n")
            continue
        if not user_text:
            continue

        history.extend(encode_chat_prompt(user_text))
        history = trim_history(history)

        reply_ids = generate(model, history, device, gen_cfg)
        reply = enc.decode(reply_ids).strip()
        last_user, last_reply = user_text, reply

        # record the full assistant turn (with eot) so the template stays intact
        history.extend(reply_ids + EOT_IDS + NEWLINE_IDS)
        history = trim_history(history)

        console.print(f"[white]{reply}[/]\n")


if __name__ == "__main__":
    main()
