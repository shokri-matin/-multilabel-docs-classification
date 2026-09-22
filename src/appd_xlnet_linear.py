from pathlib import Path
import csv

import numpy as np
import torch
import torch.nn as nn

from tqdm import tqdm
from transformers import XLNetTokenizer, XLNetModel

from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    hamming_loss,
)


# ============================================================
# Configuration
# ============================================================

TRAIN_CSV = Path(
    "output/aapd/train_closeness_no_branch_top100_nodes25.csv"
)

TEST_CSV = Path(
    "output/aapd/test_closeness_no_branch_top100_nodes25_1.csv"
)

MODEL_NAME = "xlnet-base-cased"

MODEL_DIR = Path(
    "output/models/xlnet_aapd"
)

MODEL_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# Training Configuration
# ============================================================

BATCH_SIZE = 8

EPOCHS = 10

LEARNING_RATE = 1e-3

THRESHOLD = 0.25

MAX_LENGTH = 256

MAX_TRAIN_BATCHES = 5
# Example for testing:
# MAX_TRAIN_BATCHES = 5


# ============================================================
# Device
# ============================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("=" * 70)
print("DEVICE:", DEVICE)
print("=" * 70)


# ============================================================
# Load XLNet Tokenizer
# ============================================================

print("\nLoading XLNet tokenizer...")

tokenizer = XLNetTokenizer.from_pretrained(
    MODEL_NAME
)

print("XLNet tokenizer loaded successfully.")


# ============================================================
# Read CSV
# ============================================================

def read_csv(path):
    rows = []

    with open(
        path,
        "r",
        encoding="utf-8",
        newline=""
    ) as f:

        reader = csv.DictReader(f)

        for row in reader:

            text = row["sec"]

            labels = [
                x.strip()
                for x in row["label"].split("|")
                if x.strip()
            ]

            rows.append(
                {
                    "docid": row["docid"],
                    "text": text,
                    "labels": labels,
                }
            )

    return rows


print("\nLoading training data...")

train_data = read_csv(TRAIN_CSV)

print(
    "Training documents:",
    len(train_data)
)


print("\nLoading test data...")

test_data = read_csv(TEST_CSV)

print(
    "Test documents:",
    len(test_data)
)


# ============================================================
# Build Label Vocabulary
# ============================================================

all_labels = set()

for row in train_data:
    all_labels.update(row["labels"])


labels = sorted(all_labels)

label_to_id = {
    label: i
    for i, label in enumerate(labels)
}

id_to_label = {
    i: label
    for label, i in label_to_id.items()
}

NUM_LABELS = len(labels)

print("\nNumber of labels:", NUM_LABELS)


# ============================================================
# Dataset
# ============================================================

class TextDataset(torch.utils.data.Dataset):

    def __init__(
        self,
        data,
        tokenizer,
        label_to_id,
        max_length
    ):

        self.data = data
        self.tokenizer = tokenizer
        self.label_to_id = label_to_id
        self.max_length = max_length

    def __len__(self):

        return len(self.data)

    def __getitem__(self, index):

        row = self.data[index]

        text = row["text"]

        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )

        input_ids = encoding["input_ids"].squeeze(0)

        attention_mask = encoding["attention_mask"].squeeze(0)

        target = torch.zeros(
            len(self.label_to_id),
            dtype=torch.float32
        )

        for label in row["labels"]:

            if label in self.label_to_id:

                target[
                    self.label_to_id[label]
                ] = 1.0

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": target
        }


# ============================================================
# Create Datasets
# ============================================================

print("\nCreating datasets...")

train_dataset = TextDataset(
    train_data,
    tokenizer,
    label_to_id,
    MAX_LENGTH
)

test_dataset = TextDataset(
    test_data,
    tokenizer,
    label_to_id,
    MAX_LENGTH
)


# ============================================================
# DataLoader
# ============================================================

train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

test_loader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)


print("Train batches:", len(train_loader))
print("Test batches:", len(test_loader))


# ============================================================
# XLNet Classifier
# ============================================================

class XLNetClassifier(nn.Module):

    def __init__(
        self,
        model_name,
        num_labels
    ):
        super().__init__()

        self.xlnet = XLNetModel.from_pretrained(
            model_name
        )

        # Freeze XLNet
        for parameter in self.xlnet.parameters():
            parameter.requires_grad = False

        hidden_size = self.xlnet.config.d_model

        # Two-layer classification head
        self.classifier = nn.Sequential(
            nn.Linear(
                hidden_size,
                256
            ),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(
                256,
                num_labels
            )
        )

    def forward(
        self,
        input_ids,
        attention_mask
    ):

        outputs = self.xlnet(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        hidden_states = outputs.last_hidden_state

        cls_representation = hidden_states[:, -1, :]

        logits = self.classifier(
            cls_representation
        )

        return logits


# ============================================================
# Model
# ============================================================

model = XLNetClassifier(
    model_name=MODEL_NAME,
    num_labels=NUM_LABELS
).to(DEVICE)


print("\nModel:")
print(model)


# ============================================================
# Count Trainable Parameters
# ============================================================

trainable_parameters = sum(
    parameter.numel()
    for parameter in model.parameters()
    if parameter.requires_grad
)

total_parameters = sum(
    parameter.numel()
    for parameter in model.parameters()
)

print("\nTotal parameters:", total_parameters)

print(
    "Trainable parameters:",
    trainable_parameters
)


# ============================================================
# Loss
# ============================================================

criterion = nn.BCEWithLogitsLoss()


# ============================================================
# Optimizer
# ============================================================

optimizer = torch.optim.Adam(
    filter(
        lambda parameter: parameter.requires_grad,
        model.parameters()
    ),
    lr=LEARNING_RATE
)


# ============================================================
# Evaluation
# ============================================================

def evaluate():

    model.eval()

    all_predictions = []
    all_targets = []

    with torch.no_grad():

        for batch in test_loader:

            input_ids = batch[
                "input_ids"
            ].to(DEVICE)

            attention_mask = batch[
                "attention_mask"
            ].to(DEVICE)

            y = batch[
                "labels"
            ].to(DEVICE)

            logits = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )

            probabilities = torch.sigmoid(
                logits
            )

            predictions = (
                probabilities >= THRESHOLD
            ).float()

            all_predictions.append(
                predictions.cpu().numpy()
            )

            all_targets.append(
                y.cpu().numpy()
            )

    y_pred = np.vstack(
        all_predictions
    )

    y_true = np.vstack(
        all_targets
    )

    metrics = {

        "micro_f1": f1_score(
            y_true,
            y_pred,
            average="micro",
            zero_division=0
        ),

        "macro_f1": f1_score(
            y_true,
            y_pred,
            average="macro",
            zero_division=0
        ),

        "weighted_f1": f1_score(
            y_true,
            y_pred,
            average="weighted",
            zero_division=0
        ),

        "micro_precision": precision_score(
            y_true,
            y_pred,
            average="micro",
            zero_division=0
        ),

        "macro_precision": precision_score(
            y_true,
            y_pred,
            average="macro",
            zero_division=0
        ),

        "micro_recall": recall_score(
            y_true,
            y_pred,
            average="micro",
            zero_division=0
        ),

        "macro_recall": recall_score(
            y_true,
            y_pred,
            average="macro",
            zero_division=0
        ),

        "hamming_loss": hamming_loss(
            y_true,
            y_pred
        ),
    }

    return metrics


# ============================================================
# Training
# ============================================================

print("\n" + "=" * 70)
print("START TRAINING")
print("=" * 70)


for epoch in range(1, EPOCHS + 1):

    model.train()

    running_loss = 0.0

    progress = tqdm(
        enumerate(train_loader),
        total=len(train_loader),
        desc=f"Epoch {epoch}/{EPOCHS}"
    )

    for batch_idx, batch in progress:

        if (
            MAX_TRAIN_BATCHES is not None
            and batch_idx >= MAX_TRAIN_BATCHES
        ):
            break

        input_ids = batch[
            "input_ids"
        ].to(DEVICE)

        attention_mask = batch[
            "attention_mask"
        ].to(DEVICE)

        y = batch[
            "labels"
        ].to(DEVICE)

        # ----------------------------------------------------
        # Clear gradients
        # ----------------------------------------------------

        optimizer.zero_grad()

        # ----------------------------------------------------
        # Forward
        # ----------------------------------------------------

        logits = model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        # ----------------------------------------------------
        # Loss
        # ----------------------------------------------------

        loss = criterion(
            logits,
            y
        )

        # ----------------------------------------------------
        # Backward
        # ----------------------------------------------------

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

        progress.set_postfix(
            loss=loss.item()
        )

    num_batches = (
        min(
            len(train_loader),
            MAX_TRAIN_BATCHES
        )
        if MAX_TRAIN_BATCHES is not None
        else len(train_loader)
    )

    avg_loss = (
        running_loss / num_batches
    )

    print(
        f"\nEpoch {epoch}"
    )

    print(
        f"Train Loss: {avg_loss:.6f}"
    )


    # --------------------------------------------------------
    # Evaluation
    # --------------------------------------------------------

    metrics = evaluate()

    print(
        f"Micro F1:        {metrics['micro_f1']:.4f}"
    )

    print(
        f"Macro F1:        {metrics['macro_f1']:.4f}"
    )

    print(
        f"Weighted F1:     {metrics['weighted_f1']:.4f}"
    )

    print(
        f"Micro Precision: {metrics['micro_precision']:.4f}"
    )

    print(
        f"Macro Precision: {metrics['macro_precision']:.4f}"
    )

    print(
        f"Micro Recall:    {metrics['micro_recall']:.4f}"
    )

    print(
        f"Macro Recall:    {metrics['macro_recall']:.4f}"
    )

    print(
        f"Hamming Loss:    {metrics['hamming_loss']:.6f}"
    )


    # --------------------------------------------------------
    # Checkpoint
    # --------------------------------------------------------

    checkpoint_path = (
        MODEL_DIR /
        f"epoch_{epoch}.pt"
    )

    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "label_to_id": label_to_id,
            "model_name": MODEL_NAME,
            "num_labels": NUM_LABELS,
            "max_length": MAX_LENGTH,
            "threshold": THRESHOLD,
        },
        checkpoint_path
    )

    print(
        f"Checkpoint saved: {checkpoint_path}"
    )


# ============================================================
# Final Evaluation
# ============================================================

print("\n" + "=" * 70)
print("FINAL RESULTS")
print("=" * 70)

final_metrics = evaluate()

for name, value in final_metrics.items():

    print(
        f"{name:20s}: {value:.6f}"
    )
