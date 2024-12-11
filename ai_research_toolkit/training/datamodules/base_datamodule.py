import pytorch_lightning as pl
from torch.utils.data import DataLoader
from datasets import load_dataset

class BaseDataModule(pl.LightningDataModule):
    def prepare_data(self):
        pass

    def setup(self, stage=None):
        pass

    def train_dataloader(self):
        raise NotImplementedError

    def val_dataloader(self):
        raise NotImplementedError


class HFTextClassificationDataModule(BaseDataModule):
    def __init__(self, dataset_name: str, split: str, val_split: str, batch_size: int, max_length: int):
        super().__init__()
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
        ds = load_dataset(self.dataset_name)
        self.train_data = ds[self.split]
        self.val_data = ds[self.val_split]

    def set_tokenizer(self, tokenizer):
        self.tokenizer = tokenizer

    def collate_fn(self, batch):
        # Assuming a text-classification dataset with 'text' and 'label'
        texts = [x['text'] for x in batch]
        labels = [x['label'] for x in batch]
        tokenized = self.tokenizer(texts, truncation=True, padding=True, max_length=self.max_length, return_tensors='pt')
        tokenized['labels'] = torch.tensor(labels)
        return tokenized

    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size=self.batch_size, shuffle=True, collate_fn=self.collate_fn)

    def val_dataloader(self):
        return DataLoader(self.val_data, batch_size=self.batch_size, shuffle=False, collate_fn=self.collate_fn)
