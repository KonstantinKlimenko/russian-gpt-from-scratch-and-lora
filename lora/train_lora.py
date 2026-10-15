import argparse
import json
import time
from pathlib import Path

import torch
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer


MILESTONES = (500, 2_500, 5_500, 9_500)


def parse_args():
    parser = argparse.ArgumentParser(description="Обучение LoRA для исправления текста")
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "runs" / "lora",
    )
    parser.add_argument("--resume", type=Path, help="Saved adapter directory")
    parser.add_argument("--steps", type=int, default=9_500, help="Total target step")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-length", type=int, default=160)
    return parser.parse_args()


def load_records(path):
    with path.open("r", encoding="utf-8") as file:
        return [json.loads(line) for line in file]


def encode_records(records, tokenizer, max_length):
    encoded = []
    for record in records:
        prompt = f'Исходное: {record["text_corrupt"]}\nИсправленное:'
        answer = " " + record["text"]
        prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
        answer_ids = tokenizer.encode(answer, add_special_tokens=False)
        input_ids = prompt_ids + answer_ids + [tokenizer.eos_token_id]
        if len(input_ids) > max_length:
            raise ValueError(f"Example has {len(input_ids)} tokens; limit is {max_length}")
        labels = [-100] * len(prompt_ids) + answer_ids + [tokenizer.eos_token_id]
        encoded.append({"input_ids": input_ids, "labels": labels})
    return encoded


def make_collator(pad_id):
    def collate(examples):
        rows = len(examples)
        length = max(len(example["input_ids"]) for example in examples)
        input_ids = torch.full((rows, length), pad_id, dtype=torch.long)
        labels = torch.full((rows, length), -100, dtype=torch.long)
        attention_mask = torch.zeros((rows, length), dtype=torch.long)
        for row, example in enumerate(examples):
            size = len(example["input_ids"])
            input_ids[row, :size] = torch.tensor(example["input_ids"])
            labels[row, :size] = torch.tensor(example["labels"])
            attention_mask[row, :size] = 1
        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
        }

    return collate


def to_device(batch, device):
    return {name: tensor.to(device) for name, tensor in batch.items()}


def evaluate(model, loader, device):
    model.eval()
    loss_sum = 0.0
    token_count = 0
    with torch.inference_mode():
        for batch in loader:
            batch = to_device(batch, device)
            loss = model(**batch).loss
            count = int((batch["labels"][:, 1:] != -100).sum().item())
            loss_sum += loss.item() * count
            token_count += count
    return loss_sum / token_count


def save_adapter(model, optimizer, scaler, output_dir, step, val_loss):
    folder = output_dir / f"qwen3_1_7b_gec_lora_step{step}"
    if folder.exists():
        raise FileExistsError(folder)
    model.save_pretrained(folder)
    torch.save(
        {
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(),
            "steps": step,
            "val_loss": val_loss,
        },
        folder / "training_state.pt",
    )
    print("Saved:", folder)


def train_segment(model, loader, optimizer, scaler, device, start_step, steps):
    model.train()
    started = time.perf_counter()
    recent_loss = 0.0
    iterator = iter(loader)
    for number in range(1, steps + 1):
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            batch = next(iterator)
        batch = to_device(batch, device)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            loss = model(**batch).loss
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        recent_loss += loss.item()
        if number % 100 == 0:
            print("Step:", start_step + number, "Train loss:", round(recent_loss / 100, 4))
            recent_loss = 0.0
    print("Minutes:", round((time.perf_counter() - started) / 60, 1))


def main():
    args = parse_args()
    if args.steps < 1 or args.batch_size < 1 or args.max_length < 1:
        raise ValueError("steps, batch-size and max-length must be positive")
    if not torch.cuda.is_available():
        raise RuntimeError("This fp16 training script requires CUDA")
    torch.manual_seed(42)

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    train_records = load_records(args.data_dir / "gec_train.jsonl")
    val_records = load_records(args.data_dir / "gec_val.jsonl")
    collate = make_collator(tokenizer.pad_token_id)
    train_loader = DataLoader(
        encode_records(train_records, tokenizer, args.max_length),
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate,
    )
    val_loader = DataLoader(
        encode_records(val_records, tokenizer, args.max_length),
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate,
    )
    if len(train_loader) == 0 or len(val_loader) == 0:
        raise ValueError("Train and validation data must be nonempty")

    base_model = AutoModelForCausalLM.from_pretrained(
        args.model_dir, dtype=torch.float16, device_map="auto"
    )
    if args.resume is None:
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=8,
            lora_alpha=16,
            lora_dropout=0.05,
            target_modules=["q_proj", "v_proj"],
            bias="none",
        )
        model = get_peft_model(base_model, lora_config)
        state = None
        current_step = 0
    else:
        model = PeftModel.from_pretrained(base_model, args.resume, is_trainable=True)
        state = torch.load(
            args.resume / "training_state.pt", map_location="cpu", weights_only=True
        )
        current_step = int(state["steps"])
    model.config.use_cache = False
    device = next(model.parameters()).device
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=1e-4,
        weight_decay=0.01,
    )
    scaler = torch.amp.GradScaler("cuda")
    if state is not None:
        optimizer.load_state_dict(state["optimizer"])
        scaler.load_state_dict(state["scaler"])

    args.output_dir.mkdir(parents=True, exist_ok=True)
    print("Train:", len(train_records), "Val:", len(val_records))
    print("Starting step:", current_step)
    print("Val loss:", round(evaluate(model, val_loader, device), 4))

    targets = [step for step in MILESTONES if current_step < step <= args.steps]
    if current_step < args.steps and args.steps not in targets:
        targets.append(args.steps)
    for target in targets:
        train_segment(
            model, train_loader, optimizer, scaler, device, current_step, target - current_step
        )
        current_step = target
        val_loss = evaluate(model, val_loader, device)
        print("Step:", current_step, "Val loss:", round(val_loss, 4))
        save_adapter(model, optimizer, scaler, args.output_dir, current_step, val_loss)


if __name__ == "__main__":
    main()
