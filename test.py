import torch
import torch.nn as nn
import torchmetrics
import json
import yaml
import polars as pl
from dataloader import get_dataloader
from model import BiLSTMModel


def evaluate(model: nn.Module, test_loader, tag2idx, idx2tag, save_path):
    precision_metric = torchmetrics.Precision(
        task="multiclass", num_classes=len(tag2idx), average=None, ignore_index=-1
    )
    recall_metric = torchmetrics.Recall(
        task="multiclass", num_classes=len(tag2idx), average=None, ignore_index=-1
    )
    f1_metric = torchmetrics.F1Score(
        task="multiclass", num_classes=len(tag2idx), average=None, ignore_index=-1
    )

    precision_metric.reset()
    recall_metric.reset()
    f1_metric.reset()

    with torch.no_grad():
        for tokens, tags in test_loader:
            outputs = model(tokens)
            outputs = outputs.view(-1, outputs.shape[-1])
            tags = tags.view(-1)

            predictions = torch.argmax(outputs, dim=1)
            precision_metric.update(predictions, tags)
            recall_metric.update(predictions, tags)
            f1_metric.update(predictions, tags)

    precision = precision_metric.compute().tolist()
    recall = recall_metric.compute().tolist()
    f1_score = f1_metric.compute().tolist()

    # Save results to CSV using Polars
    metrics_df = pl.DataFrame(
        {
            "Label": [idx2tag[idx] for idx in range(len(tag2idx)) if idx != -1],
            "Precision": [precision[idx] for idx in range(len(tag2idx)) if idx != -1],
            "Recall": [recall[idx] for idx in range(len(tag2idx)) if idx != -1],
            "F1-Score": [f1_score[idx] for idx in range(len(tag2idx)) if idx != -1],
        }
    )
    metrics_df.write_csv(save_path)
    print(f"Results saved to {save_path}")


if __name__ == "__main__":
    # Load config
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    with open(config["word2idx_path"]) as f:
        word2idx = json.load(f)
    with open(config["tag2idx_path"]) as f:
        tag2idx = json.load(f)
    idx2tag = {v: k for k, v in tag2idx.items()}

    test_loader = get_dataloader(
        config["test_url"],
        batch_size=config["batch_size"],
        shuffle=False,
    )

    model = BiLSTMModel(
        len(word2idx),
        embedding_dim=config["embedding_dim"],
        hidden_dim=config["hidden_dim"],
        num_labels=len(tag2idx),
        pad_idx=0,
    )
    model.load_state_dict(torch.load(config["model_path"]))
    model.eval()

    evaluate(model, test_loader, tag2idx, idx2tag, config["test_save_path"])
