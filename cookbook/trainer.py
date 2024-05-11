import torch
import logging
from utils import save_experiment
from dataclasses import dataclass
import sys
import wandb
import math
import string
import os
import torch.optim as optim
import torch
import numpy as np
import datetime
import tqdm
import matplotlib.pyplot as plt
import time
import random
import pandas as pd
import logging
import torch.optim.lr_scheduler as lr_scheduler

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class BaseTrainer:
    def __init__(self, model, optimizer, loss_fn, train_dataloader, valid_dataloader, test_dataloader, n_epochs, hyperparameters=None, wandb_logger=None):
        self.model = model.to(DEVICE)
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.device = DEVICE
        self.train_dataloader = train_dataloader
        self.valid_dataloader = valid_dataloader
        self.test_dataloader = test_dataloader
        self.n_epochs = n_epochs
        self.hyperparameters = hyperparameters
        self.wandb_logger = wandb_logger
        self.lr_scheduler = None
        
    def train(self):
        raise NotImplementedError

    def train_epoch(self):
        raise NotImplementedError

    def evaluate(self):
        raise NotImplementedError

    def evaluate_epoch(self):
        raise NotImplementedError

    def get_accuracy(self, prediction, label):
        raise NotImplementedError



class VisionClassificationTrainer(BaseTrainer):
    def __init__(self, model, optimizer, loss_fn, train_dataloader, valid_dataloader, test_dataloader, n_epochs, hyperparameters=None, wandb_logger=None):
        super().__init__(model, optimizer, loss_fn, train_dataloader, valid_dataloader, test_dataloader, n_epochs, hyperparameters, wandb_logger)

    def train(self, save_model_every_n_epochs=0):
        train_losses, test_losses, accuracies = [], [], []

        for i in range(self.n_epochs):
            train_loss = self.train_epoch(self.train_dataloader)
            accuracy, test_loss = self.evaluate()
            train_losses.append(train_loss)
            test_losses.append(test_loss)
            accuracies.append(accuracy)
            logging.info(
                f"Epoch: {i+1}, Train loss: {train_loss:.4f}, Test loss: {test_loss:.4f}, Accuracy: {accuracy:.4f}"
            )
            train_test_metrics = self._make_metrics(train_loss, accuracy)
            self.wandb_logger.log_metrics(train_test_metrics)

        
    def train_epoch(self, trainloader):
        self.model.train()
        total_loss = 0
        for batch in self.train_dataloader:
            batch = [t.to(DEVICE) for t in batch]
            images, labels = batch
            self.optimizer.zero_grad()
            loss = self.loss_fn(self.model(images)[0], labels)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item() * len(images)
        return total_loss / len(self.train_dataloader.dataset)

    @torch.no_grad()
    def evaluate(self):
        self.model.eval()
        total_loss = 0
        correct = 0
        with torch.no_grad():
            for batch in self.test_dataloader:
                batch = [t.to(DEVICE) for t in batch]
                images, labels = batch

                logits, _ = self.model(images)

                loss = self.loss_fn(logits, labels)
                total_loss += loss.item() * len(images)

                predictions = torch.argmax(logits, dim=1)
                correct += torch.sum(predictions == labels).item()
        accuracy = correct / len(self.test_dataloader.dataset)
        avg_loss = total_loss / len(self.test_dataloader.dataset)
        return accuracy, avg_loss

    def _make_metrics(self, train_loss: float, test_accuracy: float) -> dict:
        return {'train_loss': train_loss, 'test_accuracy': test_accuracy}




class TextClassificationTrainer(BaseTrainer):
    def __init__(
        self,
        model,
        optimizer,
        loss_fn,
        train_dataloader,
        valid_dataloader,
        test_dataloader,
        n_epochs=4,
        hyperparameters=None,
        wandb_logger = None
    ):
        super().__init__(model, optimizer, loss_fn, train_dataloader, valid_dataloader, test_dataloader, n_epochs, hyperparameters, wandb_logger) 
        self.train_losses = []
        self.train_accs = []
        self.valid_losses = []
        self.valid_accs = []
        self.loss = 'ce'
        

    def get_accuracy(self, prediction, label):
        method_name = "_get_accuracy_" + self.loss
        if hasattr(self, method_name):
            accuracy_method = getattr(self, method_name)
            return accuracy_method(prediction, label)
        else:
            raise ValueError("Invalid loss function: {}".format(self.loss))

    def _get_accuracy_ce(self, prediction, label):
        batch_size, _ = prediction.shape
        predicted_classes = prediction.argmax(dim=-1)
        correct_predictions = predicted_classes.eq(label).sum()
        accuracy = correct_predictions / batch_size
        return accuracy

    def _get_accuracy_bce(self, prediction, label):
        batch_size = prediction.shape
        acc = torch.sum(torch.round(prediction) == label)
        accuracy = acc / batch_size[0]
        return accuracy

    def train_epoch(self):
        self.model.train()
        epoch_losses = []
        epoch_accs = []

        for batch in tqdm.tqdm(
            self.train_dataloader, desc="training...", file=sys.stdout
        ):
            ids = batch["ids"].to(self.device)
            label = batch["label"].to(self.device)
            returns = self.model(ids)
            if len(returns) == 2:
                prediction, _ = returns
            else:
                prediction = returns
            if self.loss == "bce":
                prediction = prediction.squeeze()
                label = label.float()
            loss = self.loss_fn(prediction, label)
            accuracy = self.get_accuracy(prediction, label)
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.1)
            self.optimizer.step()
            epoch_losses.append(loss.item())
            epoch_accs.append(accuracy.item())
        
        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

        return epoch_losses, epoch_accs

    def evaluate_epoch(self, split="valid"):
        self.model.eval()
        epoch_losses = []
        epoch_accs = []
        if split == "valid":
            dataloader = self.valid_dataloader
        elif split == "test":
            dataloader = self.test_dataloader

        with torch.no_grad():
            for batch in tqdm.tqdm(dataloader, desc="evaluating...", file=sys.stdout):
                ids = batch["ids"].to(self.device)
                label = batch["label"].to(self.device)
                returns = self.model(ids)
                if len(returns) == 2: # This is adhoc. TODO: fix this RNN vs Transformer
                    prediction, _ = returns
                else:
                    prediction = returns
                if self.loss == "bce":
                    prediction = prediction.squeeze()
                    label = (
                        label.float()
                    )  # TODO: Taha check why this only works when casted to float
                loss = self.loss_fn(prediction, label)
                accuracy = self.get_accuracy(prediction, label)
                epoch_losses.append(loss.item())
                epoch_accs.append(accuracy.item())

        return epoch_losses, epoch_accs

    def train(self):

        best_valid_loss = float("inf")

        for epoch in range(self.n_epochs):
            train_loss, train_acc = self.train_epoch()
            valid_loss, valid_acc = self.evaluate_epoch(split="valid")
            print(f'lr: {self.optimizer.param_groups[0]["lr"]}')
            self.train_losses.extend(train_loss)
            self.train_accs.extend(train_acc)
            self.valid_losses.extend(valid_loss)
            self.valid_accs.extend(valid_acc)

            epoch_train_loss = np.mean(train_loss)
            epoch_train_acc = np.mean(train_acc)
            epoch_valid_loss = np.mean(valid_loss)
            epoch_valid_acc = np.mean(valid_acc)

            if epoch_valid_loss < best_valid_loss:
                best_valid_loss = epoch_valid_loss
                # torch.save(self.model.state_dict(), f'{save_dir}/best_model.pt')

            print(f"epoch: {epoch+1}")
            print(
                f"train_loss: {epoch_train_loss:.3f}, train_acc: {epoch_train_acc:.3f}"
            )
            print(
                f"valid_loss: {epoch_valid_loss:.3f}, valid_acc: {epoch_valid_acc:.3f}"
            )
            wandb.log({"train_loss": epoch_train_loss, "valid_loss": epoch_valid_loss, 
                       "train_acc": epoch_train_acc, "valid_acc": epoch_valid_acc})

        logging.info("Testing the model on the test split...") 
        self.test()

    def test(self):
        test_losses, test_accs = self.evaluate_epoch(split="test")
        epoch_test_loss = np.mean(test_losses)
        epoch_test_acc = np.mean(test_accs)
        logging.info(
            f"test_loss: {epoch_test_loss:.3f}, test_acc: {epoch_test_acc:.3f}"
        )


@dataclass
class TrainerSelector:
    text_classification = TextClassificationTrainer
    vision_classification = VisionClassificationTrainer

    @classmethod
    def select_trainer(cls, task):
        if task == 'text_classification':
            return cls.text_classification
        elif task == 'vision_classification':
            return cls.vision_classification
        else:
            raise ValueError(f"Invalid task: {task}. Expected 'text_classification' or 'vision_classification'")

class Trainer:
    def __init__(
        self,
        model,
        optimizer,
        loss_fn,
        train_dataloader,
        valid_dataloader,
        test_dataloader,
        n_epochs,
        hyperparameters=None,
        wandb_logger=None,
        task='vision_classification'
    ):
        self.trainer_class = TrainerSelector.select_trainer(task)
        self.params = {
            'model': model,
            'optimizer': optimizer,
            'loss_fn': loss_fn,
            'train_dataloader': train_dataloader,
            'valid_dataloader': valid_dataloader,
            'test_dataloader': test_dataloader,
            'n_epochs': n_epochs,
            'hyperparameters': hyperparameters,
            'wandb_logger': wandb_logger
        }

    def __call__(self):
        return self.trainer_class(**self.params)