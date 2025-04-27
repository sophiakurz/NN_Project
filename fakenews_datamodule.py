import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import DistilBertTokenizer
from lightning import LightningDataModule
from sklearn.model_selection import train_test_split

# Custom Dataset class for Fake News data
class FakeNewsDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=512):
        # Save inputs
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        # Return number of samples
        return len(self.labels)

    def __getitem__(self, idx):
        # Tokenize one article (text sample)
        encoding = self.tokenizer(
            self.texts[idx],
            truncation=True,    # Truncate if longer than max_len
            padding="max_length",   # Pad shorter sequences
            max_length=self.max_len,
            return_tensors="pt"     # Return PyTorch tensors
        )
        # Return input_ids, attention_mask, and label
        return (
            encoding["input_ids"].squeeze(0),   # Shape: [max_len]
            encoding["attention_mask"].squeeze(0),  # Shape: [max_len]
            torch.tensor(self.labels[idx])  # Label tensor
        )

# LightningDataModule to organize data loading
class FakeNewsDataModule(LightningDataModule):
    def __init__(self, true_path, fake_path, batch_size=16):
        super().__init__()
        self.true_path = true_path
        self.fake_path = fake_path
        self.batch_size = batch_size
        self.tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased") # Load tokenizer once

    def prepare_data(self):
        pass  # already have local files

    def setup(self, stage=None):
        # Load true and fake datasets
        true_df = pd.read_csv(self.true_path)
        fake_df = pd.read_csv(self.fake_path)

        # Add binary labels: 1 = True News, 0 = Fake News
        true_df["label"] = 1
        fake_df["label"] = 0

        # Merge datasets and shuffle
        df = pd.concat([true_df, fake_df]).sample(frac=1).reset_index(drop=True)

        # Combine title + text into a single 'content' field
        df["content"] = df["title"] + " " + df["text"]

        # Split into training and validation sets
        train_texts, val_texts, train_labels, val_labels = train_test_split(
            df["content"], df["label"], test_size=0.2, random_state=42
        )

        # Create Dataset objects
        self.train_dataset = FakeNewsDataset(list(train_texts), list(train_labels), self.tokenizer)
        self.val_dataset = FakeNewsDataset(list(val_texts), list(val_labels), self.tokenizer)

    def train_dataloader(self):
        # Return DataLoader for training
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=4)

    def val_dataloader(self):
        # Return DataLoader for validation
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=4)
