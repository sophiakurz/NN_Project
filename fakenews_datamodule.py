import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer
from lightning import LightningDataModule
from sklearn.model_selection import train_test_split

class FakeNewsDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt"
        )
        return (
            encoding["input_ids"].squeeze(0),
            encoding["attention_mask"].squeeze(0),
            torch.tensor(self.labels[idx])
        )

class FakeNewsDataModule(LightningDataModule):
    def __init__(self, true_path, fake_path, batch_size=16):
        super().__init__()
        self.true_path = true_path
        self.fake_path = fake_path
        self.batch_size = batch_size
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    def prepare_data(self):
        pass  # already have local files

    def setup(self, stage=None):
        true_df = pd.read_csv(self.true_path)
        fake_df = pd.read_csv(self.fake_path)
        true_df["label"] = 1
        fake_df["label"] = 0

        df = pd.concat([true_df, fake_df]).sample(frac=1).reset_index(drop=True)
        df["content"] = df["title"] + " " + df["text"]

        train_texts, val_texts, train_labels, val_labels = train_test_split(
            df["content"], df["label"], test_size=0.2, random_state=42
        )

        self.train_dataset = FakeNewsDataset(list(train_texts), list(train_labels), self.tokenizer)
        self.val_dataset = FakeNewsDataset(list(val_texts), list(val_labels), self.tokenizer)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=4)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=4)
