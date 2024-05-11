import functools
import sys
import datasets
import torch
import torchtext
import numpy as np
import random
from datasets import load_dataset
from enum import Enum
import logging
from torch.utils.data import DataLoader
from abc import ABC, abstractmethod
from torch.utils.data import DataLoader
from typing import Tuple, Any
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from typing import Tuple, List



class BaseDataset(ABC):
    @abstractmethod
    def __init__(self, *args, **kwargs):
        """
        Abstract method for initializing a dataset object. 
        The specific arguments required will depend on the implementation in the subclass.
        """
        pass

    @abstractmethod
    def prepare_data(self, *args, **kwargs) -> Tuple[DataLoader, DataLoader, Any]:
        """
        Abstract method for preparing the data for training and testing. 
        This method should return a tuple containing three elements:
        - train_dataloader: DataLoader object for the training data
        - test_dataloader: DataLoader object for the testing data
        - Additional data or metadata: This could be another DataLoader object for validation data, 
          or metadata such as the classes of a dataset. The specific nature of this element will 
          depend on the implementation in the subclass.
        """
        pass

class TextDatasetProcessor:
    def __init__(
        self,
        dataset_name: str,
        seed: int = 42,
        max_length: int = 256,
        min_freq: int = 5,
        test_size: float = 0.25,
        batch_size: int = 128,
    ):
        """
        Initializes the TextDatasetProcessor object.

        Args:
            dataset_name (str): The name of the dataset to be processed.
            seed (int, optional): The seed for the random number generator. Defaults to 42.
            max_length (int, optional): The maximum length of the tokenized sequences. Defaults to 256.
            min_freq (int, optional): The minimum frequency for a token to be included in the vocabulary. Defaults to 5.
            test_size (float, optional): The proportion of the dataset to include in the test split. Defaults to 0.25.
            batch_size (int, optional): The number of samples per batch to load. Defaults to 128.
        """

        # self._set_seed(seed)
        self.max_length = max_length
        self.min_freq = min_freq
        self.test_size = test_size
        self.batch_size = batch_size
        self.dataset_name = dataset_name

            
    def _set_seed(self, seed=42):
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def _tokenize_example(self, example, tokenizer):
        tokens = tokenizer(example["text"])[: self.max_length]
        return {"tokens": tokens}

    def _numericalize_data(self, example, vocab):
        ids = [vocab[token] for token in example["tokens"]]
        return {"ids": ids}

    def _pad_sequences(self, row, padding_id, bos_index, eos_index):
        tensor = row["ids"]
        n_pad = self.max_length - len(tensor)
        tensor.extend([padding_id] * n_pad)
        tensor[0] = bos_index
        tensor[-1] = eos_index
        return {"ids": torch.tensor(tensor)}

    def _build_vocab(self, train_data):
        special_tokens = ["<unk>", "<pad>", "<eos>", "<bos>", "<mask>"]
        vocab = torchtext.vocab.build_vocab_from_iterator(
            train_data["tokens"], min_freq=self.min_freq, specials=special_tokens
        )
        logging.info(f"Vocab size: {len(vocab)}")
        unk_index = vocab["<unk>"]
        bos_index = vocab["<bos>"]
        eos_index = vocab["<eos>"]
        pad_index = vocab["<pad>"]
        mask_index = vocab["<mask>"]
        mask_ignore_ids = [unk_index, bos_index, eos_index, pad_index, mask_index]
        vocab.set_default_index(unk_index)
        self.vocab = vocab
        self.pad_index = pad_index
        # This will be used for the masked language model
        self.mask_ignore_ids = mask_ignore_ids
        return vocab, unk_index, bos_index, eos_index, pad_index

    def prepare_data(self):
        train_data, test_data = datasets.load_dataset(
            self.dataset_name, split=["train", "test"]
        )

        tokenizer = torchtext.data.utils.get_tokenizer("basic_english")

        train_data = train_data.map(
            self._tokenize_example, fn_kwargs={"tokenizer": tokenizer}
        )
        test_data = test_data.map(
            self._tokenize_example, fn_kwargs={"tokenizer": tokenizer}
        )

        train_valid_data = train_data.train_test_split(test_size=self.test_size)
        train_data = train_valid_data["train"]
        valid_data = train_valid_data["test"]

        vocab, unk_index, bos_index, eos_index, pad_index = self._build_vocab(
            train_data
        )

        train_data = train_data.map(self._numericalize_data, fn_kwargs={"vocab": vocab})
        valid_data = valid_data.map(self._numericalize_data, fn_kwargs={"vocab": vocab})
        test_data = test_data.map(self._numericalize_data, fn_kwargs={"vocab": vocab})

        train_data = train_data.map(
            self._pad_sequences,
            fn_kwargs={
                "padding_id": pad_index,
                "bos_index": bos_index,
                "eos_index": eos_index,
            },
        )
        valid_data = valid_data.map(
            self._pad_sequences,
            fn_kwargs={
                "padding_id": pad_index,
                "bos_index": bos_index,
                "eos_index": eos_index,
            },
        )
        test_data = test_data.map(
            self._pad_sequences,
            fn_kwargs={
                "padding_id": pad_index,
                "bos_index": bos_index,
                "eos_index": eos_index,
            },
        )

        columns = ["ids", "label"]
        train_data = train_data.with_format(type="torch", columns=columns)
        valid_data = valid_data.with_format(type="torch", columns=columns)
        test_data = test_data.with_format(type="torch", columns=columns)

        train_dataloader = torch.utils.data.DataLoader(
            train_data, batch_size=self.batch_size, shuffle=True
        )

        valid_dataloader = torch.utils.data.DataLoader(
            valid_data, batch_size=self.batch_size
        )
        test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=1)

        return train_dataloader, valid_dataloader, test_dataloader, vocab, pad_index



class CIFAR10Dataset(BaseDataset):
    def __init__(self, batch_size: int = 4, num_workers: int = 2, root_data_dir='./data'):
        self.train_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((32, 32)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomResizedCrop((32, 32), scale=(0.8, 1.0), ratio=(0.75, 1.3333333333333333), interpolation=2),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        
        self.test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((32, 32)),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    
        self.classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.root_data_dir = root_data_dir

    def prepare_data(self) -> Tuple[DataLoader, DataLoader, Tuple[str, ...]]:
        trainset = torchvision.datasets.CIFAR10(root=self.root_data_dir,
                                                train=True, download=True,
                                                transform=self.train_transform)
        trainloader = torch.utils.data.DataLoader(trainset,
                                                  batch_size=self.batch_size,
                                                  shuffle=True,
                                                  num_workers=self.num_workers)

        testset = torchvision.datasets.CIFAR10(root=self.root_data_dir,
                                               train=False,
                                               download=True,
                                               transform=self.test_transform)
        testloader = torch.utils.data.DataLoader(testset,
                                                 batch_size=self.batch_size,
                                                 shuffle=False,
                                                 num_workers=self.num_workers)

        return trainloader, testloader, self.classes



class IMDBDataset(BaseDataset):
    def __init__(
        self,
        dataset_name,
        seed=42,
        max_length=256,
        min_freq=5,
        test_size=0.25,
        batch_size=128,
    ):
        self.dataset_config = {
            "seed": seed,
            "max_length": max_length,
            "min_freq": min_freq,
            "test_size": test_size,
            "batch_size": batch_size,
            "dataset_name": dataset_name
        }
        self.processor = TextDatasetProcessor

    def prepare_data(self) -> Tuple[DataLoader, DataLoader, DataLoader]:
        self.train_dataloader, self.valid_dataloader, self.test_dataloader, vocab, pad_index = self.processor(**self.dataset_config).prepare_data()
        return self.train_dataloader, self.valid_dataloader, self.test_dataloader, vocab, pad_index