import torch
import torch.nn as nn
import torch.optim as optim
import torchmetrics
import yaml
from torch.utils.tensorboard import SummaryWriter
from dataloader import get_dataloader
from model import BiLSTMModel
import json


class Trainer:
    def __init__(
        self,
        model,
        train_loader,
        valid_loader,
        criterion,
        optimizer,
        num_epochs,
        log_dir,
        save_path,
        tag2idx,
    ):
        self.model = model
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.num_epochs = num_epochs
        self.writer = SummaryWriter(log_dir)
        self.metric = torchmetrics.F1Score(
            task="multiclass",
            num_classes=len(tag2idx),
            average="macro",
            ignore_index=-1,
        )
        self.best_valid_loss = float("inf")
        self.save_path = save_path

    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0
        self.metric.reset()

        for tokens, tags in self.train_loader:
            self.optimizer.zero_grad()
            outputs = self.model(tokens)
            outputs = outputs.view(-1, outputs.shape[-1])
            tags = tags.view(-1)
            loss = self.criterion(outputs, tags)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            self.metric.update(torch.argmax(outputs, dim=1), tags)

        avg_loss = total_loss / len(self.train_loader)
        avg_metric = self.metric.compute().item()

        self.writer.add_scalar("Loss/Train", avg_loss, epoch)
        self.writer.add_scalar("f1/Train", avg_metric, epoch)

        return avg_loss, avg_metric

    def validate(self, epoch):
        self.model.eval()
        total_loss = 0
        self.metric.reset()

        with torch.no_grad():
            for tokens, tags in self.valid_loader:
                outputs = self.model(tokens)
                outputs = outputs.view(-1, outputs.shape[-1])
                tags = tags.view(-1)
                loss = self.criterion(outputs, tags)

                total_loss += loss.item()
                self.metric.update(torch.argmax(outputs, dim=1), tags)

        avg_loss = total_loss / len(self.valid_loader)
        avg_metric = self.metric.compute().item()

        self.writer.add_scalar("Loss/Validation", avg_loss, epoch)
        self.writer.add_scalar("f1/Validation", avg_metric, epoch)

        return avg_loss, avg_metric

    def train(self):
        for epoch in range(self.num_epochs):
            train_loss, train_f1 = self.train_epoch(epoch)
            valid_loss, valid_f1 = self.validate(epoch)

            print(
                f"Epoch {epoch + 1}/{self.num_epochs} - Train Loss: {train_loss:.4f}, Train F1: {train_f1:.4f}, "
                f"Valid Loss: {valid_loss:.4f}, Valid F1: {valid_f1:.4f}"
            )

            # Save best model based on validation loss
            if valid_loss < self.best_valid_loss:
                self.best_valid_loss = valid_loss
                torch.save(self.model.state_dict(), self.save_path)
                print(
                    f"Model saved at epoch {epoch + 1} with validation loss: {valid_loss:.4f}"
                )

        self.writer.close()


if __name__ == "__main__":
    # Load config
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    with open(config["word2idx_path"]) as f:
        word2idx = json.load(f)
    with open(config["tag2idx_path"]) as f:
        tag2idx = json.load(f)

    train_loader = get_dataloader(
        config["train_url"],
        batch_size=config["batch_size"],
        shuffle=True,
    )
    valid_loader = get_dataloader(
        config["valid_url"],
        batch_size=config["batch_size"],
        shuffle=False,
    )

    model = BiLSTMModel(
        len(word2idx),
        config["embedding_dim"],
        config["hidden_dim"],
        len(tag2idx),
        pad_idx=0,
    )
    criterion = nn.CrossEntropyLoss(ignore_index=-1)
    optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"])

    trainer = Trainer(
        model,
        train_loader,
        valid_loader,
        criterion,
        optimizer,
        config["num_epochs"],
        config["log_dir"],
        config["model_path"],
        tag2idx,
    )
    trainer.train()
    # tensorboard --logdir runs
    # http://localhost:6006/
