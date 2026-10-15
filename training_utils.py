from pathlib import Path
from time import perf_counter

import torch
import torch.nn.functional as F


def evaluate_full(model, loader, device):
    was_training = model.training
    model.eval()

    loss_sum = 0.0
    token_count = 0

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)

            logits = model(x)
            loss_sum += F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                y.reshape(-1),
                reduction="sum",
            ).item()
            token_count += y.numel()

    model.train(was_training)
    return loss_sum / token_count

def train_steps(
        model,
        optimizer,
        train_loader,
        val_loader,
        checkpoint,
        steps,
        save_every,
        checkpoint_dir,
        name_suffix,
        device,
):
    if steps <= 0 or save_every <= 0:
        raise ValueError("steps and save_every должны быть положительными")

    if steps > len(train_loader):
        raise ValueError("Для этого запуска шагов больше, чем порций в train_loader")

    start_step = int(checkpoint["step"])
    loss_sum = torch.zeros((), device=device)
    loss_count = 0
    started_at = perf_counter()

    model.train()

    for completed, (x, y) in enumerate(train_loader, start=1):
        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad(set_to_none=True)

        logits = model(x)
        loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                y.reshape(-1),
            )

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        loss_sum += loss.detach()
        loss_count += 1

        if completed % save_every == 0 or completed == steps:
            current_step = start_step + completed
            train_loss = (loss_sum / loss_count).item()
            val_loss = evaluate_full(model, val_loader, device)

            save_path = Path(checkpoint_dir) / (
                f"small_gpt_step{current_step}_{name_suffix}.pt"
            )

            if save_path.exists():
                raise FileExistsError(save_path)

            checkpoint = {
                **checkpoint,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "step": current_step,
                "val_loss": val_loss,
            }

            torch.save(checkpoint, save_path)

            print(
                "Шаг:", current_step,
                "средний train loss:", round(train_loss, 3),
                "полный val loss:", round(val_loss, 3)
            )
            print("Сохранено:", save_path)

            loss_sum = torch.zeros((), device=device)
            loss_count = 0

        if completed == steps:
            break

    print("Новых шагов:", completed)
    print("Всего шагов:", checkpoint["step"])
    print("Минут прошло:", round((perf_counter() - started_at) / 60, 1))

    return checkpoint