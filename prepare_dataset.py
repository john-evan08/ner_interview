# create a class to load dataset and save word2idx (vocab) and label2idx
import json
from pathlib import Path
from urllib import request
import polars as pl


class DataPreparer:
    def __init__(self, train_url, label2idx_url, save_folder="data"):
        self.train_df = pl.read_parquet(train_url)
        self.label2idx = json.load(request.urlopen(label2idx_url))
        self.idx2label = {v: k for k, v in self.label2idx.items()}
        self.word2idx = {}
        self.idx2word = {}
        self.save_folder = Path(save_folder)

    def build_vocab(self, min_freq=1):
        word_counts = self.train_df.explode("tokens")["tokens"].value_counts()
        vocab = word_counts.filter(pl.col("count") >= min_freq)["tokens"].to_list()
        self.word2idx = {word: idx for idx, word in enumerate(vocab, start=2)}
        self.word2idx["<PAD>"] = 0  # padding
        self.word2idx["<UNK>"] = 1  # unknown word OOV
        self.idx2word = {idx: word for word, idx in self.word2idx.items()}
        return

    def save_mappings(self):
        with open(self.save_folder / "word2idx.json", "w") as f:
            json.dump(self.word2idx, f, indent=4)
        with open(self.save_folder / "idx2word.json", "w") as f:
            json.dump(self.idx2word, f, indent=4)
        with open(self.save_folder / "tag2idx.json", "w") as f:
            json.dump(self.label2idx, f, indent=4)
        with open(self.save_folder / "idx2tag.json", "w") as f:
            json.dump(self.idx2label, f, indent=4)

    def run(self):
        print("Building vocabulary...")
        self.build_vocab()
        print(f"Vocabulary size: {len(self.word2idx)}")
        print("Saving mappings...")
        self.save_mappings()
        print("Preparation done!")


if __name__ == "__main__":
    train_url = "https://huggingface.co/api/datasets/tner/conll2003/parquet/conll2003/train/0.parquet"
    label2idx_url = (
        "https://huggingface.co/datasets/tner/conll2003/raw/main/dataset/label.json"
    )

    preparer = DataPreparer(train_url, label2idx_url)
    preparer.run()
