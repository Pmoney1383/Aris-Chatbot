"""Stage 0 inference REPL — text continuation with the base model.

NOT a chatbot yet: the base model just continues whatever text you type.
Speaker tokens / conversation handling come after SFT.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import tiktoken
import torch
import torch.nn.functional as F
from rich.console import Console

from src.config import GenerationConfig, ModelConfig, TrainConfig
from src.model import DecoderOnlyTransformer

console = Console()


def load_model(device):
    cfg = TrainConfig()
    ckpts = sorted(cfg.checkpoint_dir.glob("ckpt_*.pt"))
    if not ckpts:
        console.print("[red]No checkpoints found in checkpoints/. "
                      "Run scripts/train.py first.[/]")
        sys.exit(1)

    ckpt_path = ckpts[-1]
    console.print(f"Loading [bold]{ckpt_path.name}[/]...")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    model_cfg = ModelConfig(**ckpt["model_config"])
    model = DecoderOnlyTransformer(model_cfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    console.print(f"Checkpoint step: {ckpt['step']}, loss: {ckpt['loss']:.4f}")
    return model


@torch.no_grad()
def generate(model, enc, prompt_ids, device, gen_cfg: GenerationConfig):
    stop_ids = enc.encode_ordinary("\n\n")
    x = torch.tensor([prompt_ids], dtype=torch.long, device=device)
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

        # stop on "\n\n" (either as a single token or as the tail of generation)
        if generated[-len(stop_ids):] == stop_ids or "\n\n" in enc.decode(generated[-2:]):
            break

    return generated


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    console.print(f"Device: {device}")

    console.print(
        "\n[bold yellow]NOTE:[/] this is a BASE model — it does text continuation, "
        "not conversation.\nType a sentence and the model will continue it. "
        "Don't expect it to answer questions or chat.\n"
        "Type 'exit' or 'quit' to leave.\n"
    )

    model = load_model(device)
    enc = tiktoken.get_encoding("gpt2")
    gen_cfg = GenerationConfig()

    while True:
        try:
            prompt = console.input("[bold cyan]prompt> [/]").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if prompt.lower() in ("exit", "quit"):
            break
        if not prompt:
            continue

        prompt_ids = enc.encode_ordinary(prompt)
        out_ids = generate(model, enc, prompt_ids, device, gen_cfg)

        console.print(f"[cyan]{prompt}[/]", end="")
        console.print(f"[green]{enc.decode(out_ids)}[/]\n")


if __name__ == "__main__":
    main()
