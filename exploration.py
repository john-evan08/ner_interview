import polars as pl
import json
from urllib import request
from pathlib import Path


class DataExploration:
    def __init__(self, train_url, valid_url, label2id_url, save_folder="plots"):
        self.train_url = train_url
        self.valid_url = valid_url
        self.label2id = json.load(request.urlopen(label2id_url))
        self.id2label = {v: k for k, v in self.label2id.items()}
        self.save_folder = Path(save_folder)
        self.log_file = self.save_folder / "logs.txt"
        self.df = self.load_data()
        self.clear_logs()

    def clear_logs(self):
        self.log_file.write_text("")

    def log(self, message):
        with open(self.log_file, "a") as f:
            f.write(message + "\n")

    def load_data(self):
        train_df = pl.read_parquet(self.train_url)
        valid_df = pl.read_parquet(self.valid_url)
        return pl.concat([train_df, valid_df])

    def count_entities(self):
        # explode tags and count
        tags_df = self.df.explode("tags")["tags"].value_counts()
        return tags_df.with_columns(
            pl.col("tags")
            .replace_strict(self.id2label, return_dtype=pl.String)
            .alias("tags")
        )

    def plot_entity_distrib(self):
        tags_df = self.count_entities()
        chart = tags_df.plot.bar(
            x="tags",
            y="count",
        )
        chart.save(self.save_folder / "entity_distribution.png")

        # remove rows with tag = "0"

        tags_df_no0 = tags_df.filter(pl.col("tags") != "O")
        chart = tags_df_no0.plot.bar(
            x="tags",
            y="count",
        )
        chart.save(self.save_folder / "entity_distribution_no0.png")
        return

    def plot_sequence_length_distrib(self):
        # count sequence length
        sequence_lengths = self.df["tokens"].list.len()
        # print min max mean
        self.log(f"Min sequence length: {sequence_lengths.min()}")
        self.log(f"Max sequence length: {sequence_lengths.max()}")
        self.log(f"Mean sequence length: {sequence_lengths.mean():.0f}")
        chart = sequence_lengths.plot.kde()
        chart.save(self.save_folder / "sequence_length_distribution.png")
        return

    def get_vocab_size(self):
        return self.df.explode("tokens")["tokens"].n_unique()

    def examples(self, n=3):
        ex_df = self.df.head(n)
        ex_df = ex_df.with_columns(
            pl.col("tags")
            .map_elements(
                lambda array: [self.id2label[i] for i in array],
                return_dtype=pl.List(pl.String),
            )
            .alias("tags")
        )

        for row in ex_df.rows(named=True):
            tokens = row["tokens"]
            tags = row["tags"]
            self.log(f"Tokens: {tokens}\nTags: {tags}\n-------")


if __name__ == "__main__":
    print("Loading data...")
    train_url = "https://huggingface.co/api/datasets/tner/conll2003/parquet/conll2003/train/0.parquet"
    valid_url = "https://huggingface.co/api/datasets/tner/conll2003/parquet/conll2003/validation/0.parquet"
    label2id_url = (
        "https://huggingface.co/datasets/tner/conll2003/raw/main/dataset/label.json"
    )
    data_explorer = DataExploration(train_url, valid_url, label2id_url)
    print("Exploring...")
    data_explorer.log(f"nb of sentences {data_explorer.df.shape[0]}")
    data_explorer.log(f"Vocab size: {data_explorer.get_vocab_size()}")
    data_explorer.log("Examples:")
    data_explorer.examples(4)
    data_explorer.plot_entity_distrib()
    data_explorer.plot_sequence_length_distrib()
    data_explorer.log("Done")
    print("Done")
