import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import DistilBertTokenizer
from lightning import LightningDataModule
from sklearn.model_selection import train_test_split

# Define custom Dataset class for Fake News
class FakeNewsDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=512):
        # Initialize dataset with text, labels, tokenizer and maximum sequence length
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        # Return the total number of samples
        return len(self.labels)

    def __getitem__(self, idx):
        # Get a single item by index
        # Tokenize the text at index idx
        encoding = self.tokenizer(
            self.texts[idx],
            truncation=True,    # Truncate to max_len if too long
            padding="max_length",   # Pad to max_len if too short
            max_length=self.max_len,
            return_tensors="pt"     # Return PyTorch tensors
        )
        # Return tokenized input_ids, attention_mask, and label
        return (
            encoding["input_ids"].squeeze(0),   # Tensor of token ids
            encoding["attention_mask"].squeeze(0),  # Tensor of attention mask
            torch.tensor(self.labels[idx])  # Tensor of label
        )

# Define LightningDataModule for managing DataLoaders
class FakeNewsDataModule(LightningDataModule):
    def __init__(self, true_path, fake_path, batch_size=16):
        super().__init__()
        self.true_path = true_path  # Path to real news dataset
        self.fake_path = fake_path  # Path to fake news dataset
        self.batch_size = batch_size    # Batch size for dataloaders
        self.tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased") # Initialize DistilBERT tokenizer

    def prepare_data(self):
        pass  # already have local files

    def setup(self, stage=None):
        # Read true and fake news CSV files
        true_df = pd.read_csv(self.true_path)
        fake_df = pd.read_csv(self.fake_path)
        
        # Assign labels: 1 for true news, 0 for fake news
        true_df["label"] = 1
        fake_df["label"] = 0

        # Concatenate both datasets and shuffle
        df = pd.concat([true_df, fake_df]).sample(frac=1).reset_index(drop=True)

        # Create a 'content' column by combining title and text
        df["content"] = df["title"] + " " + df["text"]

        # Save full dataset for later splitting
        self.df = df

    def setup_from_split(self, train_idx, val_idx):
        # Setup training and validation datasets based on split indices
        df = self.df

        # Select training texts and labels
        train_texts = df.iloc[train_idx]["content"].tolist()
        train_labels = df.iloc[train_idx]["label"].tolist()

        # Select validation texts and labels
        val_texts = df.iloc[val_idx]["content"].tolist()
        val_labels = df.iloc[val_idx]["label"].tolist()

        # Create Dataset objects
        self.train_dataset = FakeNewsDataset(train_texts, train_labels, self.tokenizer)
        self.val_dataset = FakeNewsDataset(val_texts, val_labels, self.tokenizer)


    def train_dataloader(self):
        # Return training DataLoader
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=4)

    def val_dataloader(self):
        # Return validation DataLoader
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=4)
