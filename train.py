from lightning import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger
from fakenews_classifier import FakeNewsClassifier
from fakenews_datamodule import FakeNewsDataModule
import pandas as pd
import matplotlib.pyplot as plt

if __name__ == "__main__":
    # Model + DataModule
    model = FakeNewsClassifier(n_classes=2)
    datamodule = FakeNewsDataModule("News _dataset/True.csv", "News _dataset/Fake.csv", batch_size=8)

    # Callbacks and logger
    checkpoint = ModelCheckpoint(
        monitor="val_acc", mode="max", save_top_k=1, filename="best-fakenews"
    )
    logger = CSVLogger("logs", name="fakenews_run")

    # Trainer
    trainer = Trainer(max_epochs=3, callbacks=[checkpoint], logger=logger)
    trainer.fit(model, datamodule)

    # Plot training/validation accuracy from CSV logs
    metrics_path = f"{logger.log_dir}/metrics.csv"
    df = pd.read_csv(metrics_path)

    # # Drop rows without values (Lightning logs non-epoch rows too)
    # df = df.dropna(subset=["train_acc", "val_acc"])

    # plt.figure(figsize=(8, 5))
    # plt.plot(df["epoch"], df["train_acc"], label="Train Accuracy")
    # plt.plot(df["epoch"], df["val_acc"], label="Validation Accuracy")
    # plt.xlabel("Epoch")
    # plt.ylabel("Accuracy")
    # plt.title("Train vs Validation Accuracy")
    # plt.legend()
    # plt.grid(True)
    # plt.savefig("accuracy_plot.png")
    # print("✅ Accuracy plot saved as accuracy_plot.png")
