import warnings

import torch
from torch.utils.data import DataLoader

import lightning.pytorch as pl
from datasets import load_dataset

class BaseDataModule(pl.LightningDataModule):
    def __init__(self, **kwargs):  # This will handle the kwargs intended for the DataLoader
        super().__init__()
        self.dataloader_kwargs = kwargs
        # There is an issue with the MPS when using GPU + num_workers > 0
        # We don't know what accelerator is being used though :'(
        # Maybe we can change the multiprocessing_context to 'fork' to avoid issues?
        if self.dataloader_kwargs.get('num_workers', 0) > 0 and torch.backends.mps.is_available():
            warnings.warn("MPS is enabled while num_workers > 0 -- cowarding to 'fork' multiprocessing_context. "
                          "Note that this might cause deadlocks in the child process.")
            self.dataloader_kwargs['multiprocessing_context'] = 'fork'

    def prepare_data(self):
        pass

    def setup(self, stage=None):
        pass

    def train_dataloader(self):
        raise NotImplementedError

    def val_dataloader(self):
        raise NotImplementedError


class HFTextClassificationDataModule(BaseDataModule):
    def __init__(self, dataset_name: str, split: str, val_split: str, batch_size: int, max_length: int,
                 **kwargs):
        super().__init__(**kwargs)
        self.dataset_name = dataset_name
        self.split = split
        self.val_split = val_split
        self.batch_size = batch_size
        self.max_length = max_length
        self.dataset = None
        self.tokenizer = None

    def prepare_data(self):
        # Download the dataset
        load_dataset(self.dataset_name)

    def setup(self, stage=None):
        # Note: If using slicing, e.g. "train[:1%]", load_dataset returns a Dataset (not a DatasetDict)
        self.train_data = load_dataset(self.dataset_name, split=self.split)
        self.val_data = load_dataset(self.dataset_name, split=self.val_split)

    def set_tokenizer(self, tokenizer):
        self.tokenizer = tokenizer

    def collate_fn(self, batch):
        # Assuming a text-classification dataset with 'text' and 'label'
        texts = [x['text'] for x in batch]
        labels = [x['label'] for x in batch]
        tokenized = self.tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=self.max_length,
            return_tensors='pt')
        tokenized['labels'] = torch.tensor(labels)
        return tokenized

    def train_dataloader(self):
        return DataLoader(self.train_data,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=self.collate_fn,
            **self.dataloader_kwargs)

    def val_dataloader(self):
        return DataLoader(self.val_data,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=self.collate_fn,
            **self.dataloader_kwargs)
