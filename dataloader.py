import torch
from torch.utils.data import Dataset, DataLoader
import json
import polars as pl
from torch.nn.utils.rnn import pad_sequence


class NERDataset(Dataset):
    def __init__(self, data_url, word2idx_path="data/word2idx.json"):
        self.data = pl.read_parquet(data_url)
        self.word2idx = json.load(open(word2idx_path))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        tokens = self.data[idx, "tokens"]
        tag_ids = self.data[idx, "tags"]

        token_ids = [
            self.word2idx.get(token, self.word2idx["<UNK>"]) for token in tokens
        ]

        return torch.tensor(token_ids, dtype=torch.long), torch.tensor(
            tag_ids, dtype=torch.long
        )


def collate_fn(batch):
    tokens, tags = zip(*batch)

    tokens_padded = pad_sequence(tokens, batch_first=True, padding_value=0)  # PAD is 0
    tags_padded = pad_sequence(
        tags, batch_first=True, padding_value=-1
    )  # Ignore index for loss computation

    return tokens_padded, tags_padded


def get_dataloader(data_url, batch_size=16, shuffle=True):
    dataset = NERDataset(data_url)
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn
    )


if __name__ == "__main__":
    train_url = "https://huggingface.co/api/datasets/tner/conll2003/parquet/conll2003/train/0.parquet"

    train_loader = get_dataloader(
        train_url
    )  # shuffle = true -> no overfitting on the order of the data

    # valid_loader = get_dataloader(valid_url, word2idx_path, tag2idx_path, shuffle=False)
    for batch in train_loader:
        tokens, tags = batch
        print("Token batch shape:", tokens.shape)
        print("Tag batch shape:", tags.shape)
        break
