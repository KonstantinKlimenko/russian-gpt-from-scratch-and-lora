import json
from pathlib import Path
import torch
from torch.utils.data import Dataset

def read_token_ids(path, tokenizer, eos_id, max_documents=None):
    all_ids = []

    with Path(path).open("r", encoding="utf-8") as input_file:
        for document_number, line in enumerate(input_file):
            if max_documents is not None and document_number >= max_documents:
                break

            record = json.loads(line)
            ids = tokenizer.encode(
                record["text"],
                add_special_tokens=False,
            ).ids

            all_ids.extend(ids)
            all_ids.append(eos_id)

    return torch.tensor(all_ids, dtype=torch.long)

class GPTDataset(Dataset):
    def __init__(self, token_ids, context_length, stride):
        if len(token_ids) <= context_length:
            raise ValueError("Токенов недостаточно для одного обучающего окна")

        self.token_ids = token_ids
        self.context_length = context_length
        self.starts = range(
            0,
            len(token_ids) - context_length,
            stride,
        )

    def __len__(self):
        return len(self.starts)

    def __getitem__(self, index):
        start = self.starts[index]
        end = start + self.context_length

        x = self.token_ids[start:end]
        y = self.token_ids[start + 1:end + 1]

        return x, y