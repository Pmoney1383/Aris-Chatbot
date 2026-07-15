"""DCGAN training loop for 64x64 cat images.

- BCEWithLogits with one-sided label smoothing on real labels
- R1 gradient penalty on D's real-image loss (config.r1_gamma)
- bf16 autocast on CUDA for G and D forward passes
- fixed noise grid saved every sample_every steps for visual progress
- checkpoint save/auto-resume, CSV loss logging, rich live display
- Ctrl+C saves a checkpoint before exiting
"""

import csv
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torchvision.utils import save_image
from rich.console import Console
from rich.live import Live
from rich.table import Table

from src.config import GANConfig
from src.models import Discriminator, Generator

console = Console()


def load_data(config: GANConfig) -> DataLoader:
    if not config.data_file.exists():
        console.print(f"[red]Missing {config.data_file} — run vision/scripts/prepare_cats.py first.[/red]")
        sys.exit(1)
    data = torch.load(config.data_file)
    console.print(f"Loaded {config.data_file.name}: {tuple(data.shape)}")
    return DataLoader(
        TensorDataset(data),
        batch_size=config.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=config.num_workers,
        pin_memory=True,
        persistent_workers=config.num_workers > 0,
    )


def latest_checkpoint(config: GANConfig) -> Path | None:
    ckpts = sorted(config.checkpoint_dir.glob("gan_*.pt"))
    return ckpts[-1] if ckpts else None


def save_checkpoint(config, G, D, opt_g, opt_d, step: int) -> None:
    config.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    path = config.checkpoint_dir / f"gan_{step:06d}.pt"
    torch.save({
        "generator": G.state_dict(),
        "discriminator": D.state_dict(),
        "opt_g": opt_g.state_dict(),
        "opt_d": opt_d.state_dict(),
        "step": step,
        "config": config,
    }, path)
    console.print(f"[green]Saved checkpoint {path.name}[/green]")


def make_table(step, max_steps, d_loss, g_loss, d_real, d_fake,
               d_acc_real, d_acc_fake, d_acc_overall, sps, eta_s) -> Table:
    table = Table(title="DCGAN training")
    for col in ("step", "D_loss", "G_loss", "D(real)", "D(fake)",
                "D_acc_real", "D_acc_fake", "D_acc", "steps/s", "ETA"):
        table.add_column(col, justify="right")
    eta = time.strftime("%H:%M:%S", time.gmtime(eta_s)) if sps > 0 else "--"
    table.add_row(
        f"{step}/{max_steps}", f"{d_loss:.4f}", f"{g_loss:.4f}",
        f"{d_real:.3f}", f"{d_fake:.3f}",
        f"{d_acc_real:.1%}", f"{d_acc_fake:.1%}", f"{d_acc_overall:.1%}",
        f"{sps:.2f}", eta,
    )
    return table


def main() -> None:
    config = GANConfig()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_amp = device == "cuda"
    console.print(f"Device: {device}")

    loader = load_data(config)

    G = Generator(config).to(device)
    D = Discriminator(config).to(device)
    opt_g = torch.optim.Adam(G.parameters(), lr=config.lr_g, betas=config.betas)
    opt_d = torch.optim.Adam(D.parameters(), lr=config.lr_d, betas=config.betas)
    criterion = nn.BCEWithLogitsLoss()

    step = 0
    ckpt_path = latest_checkpoint(config)
    if ckpt_path is not None:
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        G.load_state_dict(ckpt["generator"])
        D.load_state_dict(ckpt["discriminator"])
        opt_g.load_state_dict(ckpt["opt_g"])
        opt_d.load_state_dict(ckpt["opt_d"])
        step = ckpt["step"]
        console.print(f"[cyan]Resumed from {ckpt_path.name} at step {step}[/cyan]")

    # Fixed noise: the same 64 vectors every run, so sample grids are comparable.
    fixed_noise = torch.randn(64, config.z_dim, device=device,
                              generator=torch.Generator(device=device).manual_seed(1337))

    config.samples_dir.mkdir(parents=True, exist_ok=True)
    config.log_dir.mkdir(parents=True, exist_ok=True)
    log_path = config.log_dir / "gan_loss.csv"
    log_file = open(log_path, "a", newline="")
    log_writer = csv.writer(log_file)
    if log_path.stat().st_size == 0:
        log_writer.writerow(["step", "d_loss", "g_loss", "d_real", "d_fake",
                              "d_acc_real", "d_acc_fake", "d_acc_overall"])

    console.print(
        "[dim]D accuracy note: healthy training oscillates around 50-80%. "
        "Pinned at 100% means D is overpowering G; collapsed to ~50% means "
        "G has fully fooled D (can also indicate mode collapse).[/dim]"
    )

    def save_samples(at_step: int) -> None:
        G.eval()
        with torch.no_grad():
            fake = G(fixed_noise).float().cpu()
        G.train()
        save_image(fake, config.samples_dir / f"step_{at_step:06d}.png",
                   nrow=8, normalize=True, value_range=(-1, 1))

    d_loss_v = g_loss_v = d_real_v = d_fake_v = 0.0
    d_acc_real_v = d_acc_fake_v = d_acc_overall_v = 0.0
    t0 = time.time()
    step0 = step
    data_iter = iter(loader)

    try:
        with Live(console=console, refresh_per_second=4) as live:
            while step < config.max_steps:
                try:
                    (real,) = next(data_iter)
                except StopIteration:
                    data_iter = iter(loader)
                    (real,) = next(data_iter)
                real = real.to(device, non_blocking=True)
                b = real.size(0)

                real_labels = torch.full((b,), config.label_smooth, device=device)
                fake_labels = torch.zeros(b, device=device)

                # ---- D step: real (smoothed labels + R1 penalty) + fake (detached) ----
                opt_d.zero_grad(set_to_none=True)
                real.requires_grad_(True)  # needed for the R1 grad w.r.t. inputs
                with torch.autocast(device, dtype=torch.bfloat16, enabled=use_amp):
                    z = torch.randn(b, config.z_dim, device=device)
                    fake = G(z)
                    real_logits = D(real)
                    fake_logits = D(fake.detach())
                    d_loss = criterion(real_logits, real_labels) + criterion(fake_logits, fake_labels)
                # R1: gamma * mean(||grad_x D(x)||^2) on real images only.
                # create_graph so the penalty itself is differentiated in d_loss.backward().
                (r1_grad,) = torch.autograd.grad(real_logits.sum(), real, create_graph=True)
                r1_penalty = r1_grad.float().pow(2).flatten(1).sum(dim=1).mean()
                d_loss = d_loss + config.r1_gamma * r1_penalty
                d_loss.backward()
                opt_d.step()

                # ---- G step: fresh fake batch, wants D to say 1.0 ----
                opt_g.zero_grad(set_to_none=True)
                with torch.autocast(device, dtype=torch.bfloat16, enabled=use_amp):
                    z = torch.randn(b, config.z_dim, device=device)
                    fake = G(z)
                    g_logits = D(fake)
                    g_loss = criterion(g_logits, torch.ones(b, device=device))
                g_loss.backward()
                opt_g.step()

                step += 1
                d_loss_v, g_loss_v = d_loss.item(), g_loss.item()
                with torch.no_grad():
                    real_probs = torch.sigmoid(real_logits).float()
                    fake_probs = torch.sigmoid(fake_logits).float()
                    d_real_v = real_probs.mean().item()
                    d_fake_v = fake_probs.mean().item()
                    d_acc_real_v = (real_probs > 0.5).float().mean().item()
                    d_acc_fake_v = (fake_probs < 0.5).float().mean().item()
                    d_acc_overall_v = (d_acc_real_v + d_acc_fake_v) / 2

                log_writer.writerow([step, f"{d_loss_v:.4f}", f"{g_loss_v:.4f}",
                                     f"{d_real_v:.4f}", f"{d_fake_v:.4f}",
                                     f"{d_acc_real_v:.4f}", f"{d_acc_fake_v:.4f}",
                                     f"{d_acc_overall_v:.4f}"])

                sps = (step - step0) / max(time.time() - t0, 1e-9)
                eta_s = (config.max_steps - step) / max(sps, 1e-9)
                live.update(make_table(step, config.max_steps, d_loss_v, g_loss_v,
                                       d_real_v, d_fake_v, d_acc_real_v, d_acc_fake_v,
                                       d_acc_overall_v, sps, eta_s))

                if step % config.sample_every == 0:
                    save_samples(step)
                if step % config.ckpt_every == 0:
                    save_checkpoint(config, G, D, opt_g, opt_d, step)
                    log_file.flush()
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted — saving checkpoint...[/yellow]")
        save_checkpoint(config, G, D, opt_g, opt_d, step)
    finally:
        log_file.close()

    if step >= config.max_steps:
        if step % config.ckpt_every != 0:
            save_checkpoint(config, G, D, opt_g, opt_d, step)
        if step % config.sample_every != 0:
            save_samples(step)
        console.print("[green]Training complete.[/green]")


if __name__ == "__main__":
    main()
