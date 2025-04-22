from lightning import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger
from classifier import FakeNewsClassifier
from datamodule import FakeNewsDataModule
import pandas as pd

if __name__ == "__main__":
    # Model + DataModule (use your cleaned dataset)
    model = FakeNewsClassifier(n_classes=2)
    datamodule = FakeNewsDataModule("cleaned_news.csv", batch_size=8)

    # Checkpoint callback
    checkpoint = ModelCheckpoint(
        monitor="val_acc", mode="max", save_top_k=1, filename="best-fakenews"
    )

    # Logging
    logger = CSVLogger("distilbert_logs", name="fakenews_run")

    # Trainer
    trainer = Trainer(
        max_epochs=3,
        callbacks=[checkpoint],
        logger=logger
    )

    # Train the model
    trainer.fit(model, datamodule)

    # Load metrics and plot
    metrics_path = f"{logger.log_dir}/metrics.csv"