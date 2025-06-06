from sklearn.model_selection import StratifiedKFold
from lightning import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger
from fakenews_classifier import FakeNewsClassifier
from fakenews_datamodule import FakeNewsDataModule
import pandas as pd

# Load full dataset once
true_df = pd.read_csv("News _dataset/True.csv")
fake_df = pd.read_csv("News _dataset/Fake.csv")

# Assign labels: 1 for real news, 0 for fake news
true_df["label"] = 1
fake_df["label"] = 0

# Combine and shuffle datasets
df = pd.concat([true_df, fake_df]).sample(frac=1).reset_index(drop=True)

# Create 'content' by concatenating title and text
df["content"] = df["title"] + " " + df["text"]

# Prepare features (X) and labels (y)
X = df["content"].tolist()
y = df["label"].tolist()

# K-fold Cross-validation
# Initialize Stratified K-Fold Cross-Validation (preserving class distribution across folds)
skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

# Cross-validation loop
for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    print(f"Fold {fold+1}")

    # Create model and datamodule
    model = FakeNewsClassifier(n_classes=2)
    datamodule = FakeNewsDataModule("News _dataset/True.csv", "News _dataset/Fake.csv", batch_size=8)
    datamodule.setup()  # loads full df
    datamodule.setup_from_split(train_idx, val_idx) # Setup datasets for current fold split

    # Setup checkpointing to save the best model based on validation accuracy
    checkpoint = ModelCheckpoint(
        monitor="val_acc", mode="max", save_top_k=1,
        filename=f"fold{fold+1}-best"
    )
    logger = CSVLogger("logs", name=f"fold{fold+1}")    # Setup logger to save metrics to CSV

    # Initialize trainer with callbacks and logger
    trainer = Trainer(
        max_epochs=3,
        callbacks=[checkpoint],
        logger=logger
    )
    trainer.fit(model, datamodule)  # Train the model on the current fold

# Saving metrics
# Gather all fold logs
fold_logs = []
for fold in range(1, 6):
    log_path = f"logs/fold{fold}/metrics.csv"
    df = pd.read_csv(log_path)

    # Get only rows with epoch-level logs
    df_fold = df[["epoch", "train_acc_epoch", "val_acc"]].dropna()
    df_fold = df_fold.groupby("epoch").mean().reset_index() # Aggregate by epoch (should already be one row per epoch)
    df_fold.columns = ["epoch", f"train_acc_fold{fold}", f"val_acc_fold{fold}"]     # Rename columns to indicate fold
    fold_logs.append(df_fold)   # Save to list
    print(f"Finished Fold {fold+1}")

# Merge all folds on epoch
from functools import reduce
df_merged = reduce(lambda left, right: pd.merge(left, right, on="epoch"), fold_logs)

# Compute mean train and validation accuracy across folds for each epoch
train_cols = [col for col in df_merged.columns if "train_acc_fold" in col]
val_cols = [col for col in df_merged.columns if "val_acc_fold" in col]
df_merged["train_acc_mean"] = df_merged[train_cols].mean(axis=1)
df_merged["val_acc_mean"] = df_merged[val_cols].mean(axis=1)

# Keep only epoch, mean train acc, mean val acc
df_final = df_merged[["epoch", "train_acc_mean", "val_acc_mean"]]
df_final = df_final.sort_values("epoch")

# Save the averaged metrics across folds
df_final.to_csv("metrics.csv", index=False)
print("Saved average cross-validation metrics to metrics.csv")
