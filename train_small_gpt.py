import argparse
from pathlib import Path

import torch
from tokenizers import Tokenizer
from torch.utils.data import ConcatDataset, DataLoader

from data_utils import GPTDataset, read_token_ids
from model import SmallGPT
from training_utils import train_steps


CONFIG = {
    "vocab_size": 16_000,
    "context_length": 256,
    "width": 512,
    "heads": 8,
    "layers": 10,
}
LR_CHANGE_STEP = 200_000


def parse_args():
    project_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Обучение небольшой GPT на русской прозе")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument(
        "--tokenizer",
        type=Path,
        default=project_dir / "data" / "proza_bpe_16k.json",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=project_dir / "runs" / "scratch"
    )
    parser.add_argument("--resume", type=Path, help="Checkpoint from an earlier run")
    parser.add_argument("--steps", type=int, default=300_000, help="Total target step")
    parser.add_argument("--save-every", type=int, default=50_000)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-train-documents", type=int, default=None)
    parser.add_argument("--max-val-documents", type=int, default=None)
    return parser.parse_args()


def make_loaders(args, tokenizer, eos_id):
    train_files = (
        args.data_dir / "proza_train.jsonl",
        args.data_dir / "proza_train_shards_001_003.jsonl",
    )
    val_file = args.data_dir / "proza_val.jsonl"
    for path in (*train_files, val_file):
        if not path.is_file():
            raise FileNotFoundError(path)

    train_parts = []
    for path in train_files:
        ids = read_token_ids(
            path, tokenizer, eos_id, max_documents=args.max_train_documents
        )
        train_parts.append(
            GPTDataset(
                ids,
                CONFIG["context_length"],
                stride=CONFIG["context_length"],
            )
        )

    val_ids = read_token_ids(
        val_file, tokenizer, eos_id, max_documents=args.max_val_documents
    )
    val_dataset = GPTDataset(
        val_ids, CONFIG["context_length"], stride=CONFIG["context_length"]
    )
    train_dataset = ConcatDataset(train_parts)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=0,
    )
    print("Train windows:", len(train_dataset), "Val windows:", len(val_dataset))
    return train_loader, val_loader


def main():
    args = parse_args()
    if args.steps < 1 or args.save_every < 1 or args.batch_size < 1:
        raise ValueError("steps, save-every and batch-size must be positive")

    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = Tokenizer.from_file(str(args.tokenizer))
    eos_id = tokenizer.token_to_id("<|endoftext|>")
    if tokenizer.get_vocab_size() != CONFIG["vocab_size"] or eos_id is None:
        raise ValueError("The tokenizer does not match the GPT configuration")
    train_loader, val_loader = make_loaders(args, tokenizer, eos_id)

    model = SmallGPT(**CONFIG).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    checkpoint = {"step": 0, "config": CONFIG}
    if args.resume is not None:
        checkpoint = torch.load(args.resume, map_location=device, weights_only=True)
        if checkpoint["config"] != CONFIG:
            raise ValueError("Checkpoint configuration does not match this model")
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])

    args.output_dir.mkdir(parents=True, exist_ok=True)
    print("Device:", device)
    print("Parameters:", sum(p.numel() for p in model.parameters()))
    print("Starting step:", checkpoint["step"])

    while checkpoint["step"] < args.steps:
        current_step = checkpoint["step"]
        lr = 3e-4 if current_step < LR_CHANGE_STEP else 1e-4
        for group in optimizer.param_groups:
            group["lr"] = lr

        next_boundary = min(
            args.steps,
            current_step + args.save_every,
            LR_CHANGE_STEP if current_step < LR_CHANGE_STEP else args.steps,
        )
        chunk_steps = next_boundary - current_step
        if chunk_steps > len(train_loader):
            raise ValueError("A training chunk exceeds one DataLoader pass")

        checkpoint = train_steps(
            model=model,
            optimizer=optimizer,
            train_loader=train_loader,
            val_loader=val_loader,
            checkpoint=checkpoint,
            steps=chunk_steps,
            save_every=chunk_steps,
            checkpoint_dir=args.output_dir,
            name_suffix="v4_4shards",
            device=device,
        )


if __name__ == "__main__":
    main()
