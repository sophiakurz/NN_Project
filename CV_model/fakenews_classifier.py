import torch
import torch.nn as nn
from transformers import DistilBertTokenizer, DistilBertModel
import torchmetrics
from lightning import LightningModule

# Define LightningModule for Fake News Classification using DistilBERT
class FakeNewsClassifier(LightningModule):
    def __init__(self, n_classes=2, lr=1e-5):
        super().__init__()
        self.save_hyperparameters() # Saves hyperparameters like lr, n_classes automatically
        # Load pre-trained DistilBERT model and tokenizer
        self.tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
        self.bert = DistilBertModel.from_pretrained("distilbert-base-uncased")
        # Classification head: linear layer that maps hidden state to number of classes
        self.classifier = nn.Linear(self.bert.config.hidden_size, n_classes)
        # Dropout to prevent overfitting
        self.dropout = nn.Dropout(0.3)
        # Loss function for classification
        self.loss_fn = nn.CrossEntropyLoss()
        # Learning rate (will be used by optimizer)
        self.lr = lr

        # Metrics to track during training and validation
        self.train_acc = torchmetrics.Accuracy(task="multiclass", num_classes=n_classes)
        self.val_acc = torchmetrics.Accuracy(task="multiclass", num_classes=n_classes)
        self.val_f1 = torchmetrics.F1Score(task="multiclass", num_classes=n_classes)

    def forward(self, input_ids, attention_mask):
        # Forward pass through DistilBERT
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        # Extract [CLS] token's embedding (first token) as sentence representation
        cls_embedding = outputs.last_hidden_state[:, 0, :]  # <== CLS token
        # Apply dropout
        cls_embedding = self.dropout(cls_embedding)
        # Pass through classification head to get logits
        return self.classifier(cls_embedding)

    def training_step(self, batch, batch_idx):
        # Unpack batch into input_ids, attention_mask, and labels
        input_ids, attention_mask, labels = batch
         # Forward pass
        logits = self(input_ids, attention_mask)
        # Compute classification loss
        loss = self.loss_fn(logits, labels)
        # Generate predictions from logits
        preds = torch.argmax(logits, dim=1)
        # Update training accuracy metric
        self.train_acc(preds, labels)
        # Log training loss and accuracy (at the end of each epoch)
        self.log("train_loss", loss, on_step=False, on_epoch=True)
        self.log("train_acc", self.train_acc, on_epoch=True)
        # Return loss for backpropagation
        return loss

    def validation_step(self, batch, batch_idx):
        # Unpack batch into input_ids, attention_mask, and labels
        input_ids, attention_mask, labels = batch
        # Forward pass
        logits = self(input_ids, attention_mask)
        # Compute validation loss
        loss = self.loss_fn(logits, labels)
        # Generate predictions
        preds = torch.argmax(logits, dim=1)
         # Update validation metrics (accuracy and F1 score)
        self.val_acc(preds, labels)
        self.val_f1(preds, labels)
        # Log validation loss, accuracy, and F1 score
        self.log("val_loss", loss, on_step=False, on_epoch=True)
        self.log("val_acc", self.val_acc, on_epoch=True)
        self.log("val_f1", self.val_f1, on_epoch=True)

    def configure_optimizers(self):
        # Define optimizer (AdamW is recommended for transformer models)
        return torch.optim.AdamW(self.parameters(), lr=self.lr)
