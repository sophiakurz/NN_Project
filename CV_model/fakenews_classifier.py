import torch
import torch.nn as nn
from transformers import DistilBertTokenizer, DistilBertModel
import torchmetrics
from lightning import LightningModule

class FakeNewsClassifier(LightningModule):
    def __init__(self, n_classes=2, lr=1e-5):
        super().__init__()
        self.save_hyperparameters()
        self.tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
        self.bert = DistilBertModel.from_pretrained("distilbert-base-uncased")
        self.classifier = nn.Linear(self.bert.config.hidden_size, n_classes)
        self.dropout = nn.Dropout(0.3)
        self.loss_fn = nn.CrossEntropyLoss()
        self.lr = lr

        self.train_acc = torchmetrics.Accuracy(task="multiclass", num_classes=n_classes)
        self.val_acc = torchmetrics.Accuracy(task="multiclass", num_classes=n_classes)
        self.val_f1 = torchmetrics.F1Score(task="multiclass", num_classes=n_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_embedding = outputs.last_hidden_state[:, 0, :]  # <== CLS token
        cls_embedding = self.dropout(cls_embedding)
        return self.classifier(cls_embedding)

    def training_step(self, batch, batch_idx):
        input_ids, attention_mask, labels = batch
        logits = self(input_ids, attention_mask)
        loss = self.loss_fn(logits, labels)
        preds = torch.argmax(logits, dim=1)
        self.train_acc(preds, labels)
        self.log("train_loss", loss, on_step=False, on_epoch=True)
        self.log("train_acc", self.train_acc, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        input_ids, attention_mask, labels = batch
        logits = self(input_ids, attention_mask)
        loss = self.loss_fn(logits, labels)
        preds = torch.argmax(logits, dim=1)
        self.val_acc(preds, labels)
        self.val_f1(preds, labels)
        self.log("val_loss", loss, on_step=False, on_epoch=True)
        self.log("val_acc", self.val_acc, on_epoch=True)
        self.log("val_f1", self.val_f1, on_epoch=True)

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr)
