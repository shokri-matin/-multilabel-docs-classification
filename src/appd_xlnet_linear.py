from pathlib import Path
import csv

import numpy as np
import torch
import torch.nn as nn

import matplotlib.pyplot as plt

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

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# Training Configuration
# ============================================================

BATCH_SIZE = 8

EPOCHS = 10

LEARNING_RATE = 1e-3

THRESHOLD = 0.25

MAX_LENGTH = 256

MAX_TRAIN_BATCHES = 5


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

train_data = read_csv(
    TRAIN_CSV
)

print(
    "Training documents:",
    len(train_data)
)


print("\nLoading test data...")

test_data = read_csv(
    TEST_CSV
)

print(
    "Test documents:",
    len(test_data)
)


# ============================================================
# Build Label Vocabulary
# ============================================================

all_labels = set()

for row in train_data:

    all_labels.update(
        row["labels"]
    )


labels = sorted(
    all_labels
)

label_to_id = {
    label: i
    for i, label in enumerate(labels)
}

id_to_label = {
    i: label
    for label, i in label_to_id.items()
}

NUM_LABELS = len(labels)

print(
    "\nNumber of labels:",
    NUM_LABELS
)


# ============================================================
# Dataset
# ============================================================

class TextDataset(
    torch.utils.data.Dataset
):

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

    def __getitem__(
        self,
        index
    ):

        row = self.data[index]

        text = row["text"]

        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )

        input_ids = encoding[
            "input_ids"
        ].squeeze(0)

        attention_mask = encoding[
            "attention_mask"
        ].squeeze(0)

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


print(
    "Train batches:",
    len(train_loader)
)

print(
    "Test batches:",
    len(test_loader)
)


# ============================================================
# XLNet Classifier
# ============================================================

class XLNetClassifier(
    nn.Module
):

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

            nn.Dropout(
                0.1
            ),

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

        cls_representation = hidden_states[
            :, -1, :
        ]

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

print(
    "\nTotal parameters:",
    total_parameters
)

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
        lambda parameter:
        parameter.requires_grad,
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

    return (
        metrics,
        y_true,
        y_pred
    )


# ============================================================
# Save Metrics CSV
# ============================================================

metrics_csv_path = (
    MODEL_DIR /
    "training_metrics.csv"
)

metrics_history = []

metrics_csv_fields = [
    "epoch",
    "train_loss",
    "micro_f1",
    "macro_f1",
    "weighted_f1",
    "micro_precision",
    "macro_precision",
    "micro_recall",
    "macro_recall",
    "hamming_loss",
]


def save_metrics_csv():

    with open(
        metrics_csv_path,
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=metrics_csv_fields
        )

        writer.writeheader()

        writer.writerows(
            metrics_history
        )


# ============================================================
# Checkpoint Function
# ============================================================

def save_checkpoint(
    path,
    epoch,
    metrics,
    train_loss
):

    torch.save(
        {
            "epoch": epoch,

            "model_state_dict":
                model.state_dict(),

            "optimizer_state_dict":
                optimizer.state_dict(),

            "label_to_id":
                label_to_id,

            "id_to_label":
                id_to_label,

            "model_name":
                MODEL_NAME,

            "num_labels":
                NUM_LABELS,

            "max_length":
                MAX_LENGTH,

            "threshold":
                THRESHOLD,

            "train_loss":
                train_loss,

            "metrics":
                metrics,
        },
        path
    )


# ============================================================
# Training
# ============================================================

print("\n" + "=" * 70)
print("START TRAINING")
print("=" * 70)


best_macro_f1 = -float("inf")

best_epoch = None

best_model_path = (
    MODEL_DIR /
    "best_model.pt"
)


for epoch in range(
    1,
    EPOCHS + 1
):

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
            loss=f"{loss.item():.4f}"
        )


    # ========================================================
    # Average Training Loss
    # ========================================================

    num_batches = (
        min(
            len(train_loader),
            MAX_TRAIN_BATCHES
        )
        if MAX_TRAIN_BATCHES is not None
        else len(train_loader)
    )

    avg_loss = (
        running_loss /
        num_batches
    )


    # ========================================================
    # Evaluation
    # ========================================================

    (
        metrics,
        y_true,
        y_pred
    ) = evaluate()


    # ========================================================
    # Store Metrics
    # ========================================================

    epoch_metrics = {

        "epoch": epoch,

        "train_loss": avg_loss,

        "micro_f1":
            metrics["micro_f1"],

        "macro_f1":
            metrics["macro_f1"],

        "weighted_f1":
            metrics["weighted_f1"],

        "micro_precision":
            metrics["micro_precision"],

        "macro_precision":
            metrics["macro_precision"],

        "micro_recall":
            metrics["micro_recall"],

        "macro_recall":
            metrics["macro_recall"],

        "hamming_loss":
            metrics["hamming_loss"],
    }

    metrics_history.append(
        epoch_metrics
    )

    save_metrics_csv()


    # ========================================================
    # Print Epoch Metrics
    # ========================================================

    print(
        "\n" + "-" * 70
    )

    print(
        f"Epoch {epoch}/{EPOCHS}"
    )

    print(
        f"Train Loss:       {avg_loss:.6f}"
    )

    print(
        f"Micro F1:         {metrics['micro_f1']:.4f}"
    )

    print(
        f"Macro F1:         {metrics['macro_f1']:.4f}"
    )

    print(
        f"Weighted F1:      {metrics['weighted_f1']:.4f}"
    )

    print(
        f"Micro Precision:  {metrics['micro_precision']:.4f}"
    )

    print(
        f"Macro Precision:  {metrics['macro_precision']:.4f}"
    )

    print(
        f"Micro Recall:     {metrics['micro_recall']:.4f}"
    )

    print(
        f"Macro Recall:     {metrics['macro_recall']:.4f}"
    )

    print(
        f"Hamming Loss:     {metrics['hamming_loss']:.6f}"
    )


    # ========================================================
    # Save Epoch Checkpoint
    # ========================================================

    checkpoint_path = (
        MODEL_DIR /
        f"epoch_{epoch}.pt"
    )

    save_checkpoint(
        checkpoint_path,
        epoch,
        metrics,
        avg_loss
    )

    print(
        f"Checkpoint saved: {checkpoint_path}"
    )


    # ========================================================
    # Save Best Model
    # ========================================================

    current_macro_f1 = (
        metrics["macro_f1"]
    )

    if current_macro_f1 > best_macro_f1:

        best_macro_f1 = (
            current_macro_f1
        )

        best_epoch = epoch

        save_checkpoint(
            best_model_path,
            epoch,
            metrics,
            avg_loss
        )

        print(
            f"*** NEW BEST MODEL ***"
        )

        print(
            f"Best Macro F1: "
            f"{best_macro_f1:.6f}"
        )

        print(
            f"Best model saved: "
            f"{best_model_path}"
        )


# ============================================================
# Load Best Model
# ============================================================

print("\n" + "=" * 70)
print("LOADING BEST MODEL")
print("=" * 70)

best_checkpoint = torch.load(
    best_model_path,
    map_location=DEVICE
)

model.load_state_dict(
    best_checkpoint[
        "model_state_dict"
    ]
)

best_epoch = best_checkpoint[
    "epoch"
]

print(
    f"Best epoch: {best_epoch}"
)

print(
    f"Best Macro F1: "
    f"{best_checkpoint['metrics']['macro_f1']:.6f}"
)


# ============================================================
# Final Evaluation Using Best Model
# ============================================================

print("\n" + "=" * 70)
print("FINAL RESULTS - BEST MODEL")
print("=" * 70)

(
    final_metrics,
    y_true,
    y_pred
) = evaluate()


for name, value in final_metrics.items():

    print(
        f"{name:20s}: {value:.6f}"
    )


# ============================================================
# Per-Label F1
# ============================================================

print("\n" + "=" * 70)
print("PER-LABEL F1")
print("=" * 70)


per_label_f1 = f1_score(
    y_true,
    y_pred,
    average=None,
    zero_division=0
)


per_label_precision = precision_score(
    y_true,
    y_pred,
    average=None,
    zero_division=0
)


per_label_recall = recall_score(
    y_true,
    y_pred,
    average=None,
    zero_division=0
)


label_results = []


for label_id in range(
    NUM_LABELS
):

    label_results.append(
        {
            "label":
                id_to_label[label_id],

            "label_id":
                label_id,

            "f1":
                per_label_f1[label_id],

            "precision":
                per_label_precision[label_id],

            "recall":
                per_label_recall[label_id],
        }
    )


# Sort by F1 descending

label_results.sort(
    key=lambda x: x["f1"],
    reverse=True
)


# ============================================================
# Print Sorted Per-Label F1
# ============================================================

print(
    f"\n{'Rank':<6}"
    f"{'Label':<20}"
    f"{'F1':<12}"
    f"{'Precision':<12}"
    f"{'Recall':<12}"
)

print(
    "-" * 65
)


for rank, result in enumerate(
    label_results,
    start=1
):

    print(
        f"{rank:<6}"
        f"{result['label']:<20}"
        f"{result['f1']:<12.4f}"
        f"{result['precision']:<12.4f}"
        f"{result['recall']:<12.4f}"
    )


# ============================================================
# Save Per-Label Results
# ============================================================

per_label_csv = (
    MODEL_DIR /
    "per_label_metrics.csv"
)


with open(
    per_label_csv,
    "w",
    newline="",
    encoding="utf-8"
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=[
            "rank",
            "label",
            "label_id",
            "f1",
            "precision",
            "recall"
        ]
    )

    writer.writeheader()

    for rank, result in enumerate(
        label_results,
        start=1
    ):

        writer.writerow(
            {
                "rank":
                    rank,

                "label":
                    result["label"],

                "label_id":
                    result["label_id"],

                "f1":
                    result["f1"],

                "precision":
                    result["precision"],

                "recall":
                    result["recall"],
            }
        )


print(
    f"\nPer-label metrics saved to:"
    f"\n{per_label_csv}"
)


# ============================================================
# Plot Metrics vs Epoch
# ============================================================

epochs = [
    item["epoch"]
    for item in metrics_history
]

train_losses = [
    item["train_loss"]
    for item in metrics_history
]

micro_f1_values = [
    item["micro_f1"]
    for item in metrics_history
]

macro_f1_values = [
    item["macro_f1"]
    for item in metrics_history
]

weighted_f1_values = [
    item["weighted_f1"]
    for item in metrics_history
]

micro_precision_values = [
    item["micro_precision"]
    for item in metrics_history
]

macro_precision_values = [
    item["macro_precision"]
    for item in metrics_history
]

micro_recall_values = [
    item["micro_recall"]
    for item in metrics_history
]

macro_recall_values = [
    item["macro_recall"]
    for item in metrics_history
]

hamming_values = [
    item["hamming_loss"]
    for item in metrics_history
]


# ============================================================
# F1 Chart
# ============================================================

plt.figure(
    figsize=(10, 6)
)

plt.plot(
    epochs,
    micro_f1_values,
    marker="o",
    label="Micro F1"
)

plt.plot(
    epochs,
    macro_f1_values,
    marker="o",
    label="Macro F1"
)

plt.plot(
    epochs,
    weighted_f1_values,
    marker="o",
    label="Weighted F1"
)

plt.xlabel(
    "Epoch"
)

plt.ylabel(
    "F1"
)

plt.title(
    "F1 Metrics vs Epoch"
)

plt.xticks(
    epochs
)

plt.grid(
    True
)

plt.legend()

plt.tight_layout()

f1_plot_path = (
    MODEL_DIR /
    "f1_vs_epoch.png"
)

plt.savefig(
    f1_plot_path,
    dpi=300
)

plt.show()

plt.close()


# ============================================================
# Precision / Recall Chart
# ============================================================

plt.figure(
    figsize=(10, 6)
)

plt.plot(
    epochs,
    micro_precision_values,
    marker="o",
    label="Micro Precision"
)

plt.plot(
    epochs,
    macro_precision_values,
    marker="o",
    label="Macro Precision"
)

plt.plot(
    epochs,
    micro_recall_values,
    marker="o",
    label="Micro Recall"
)

plt.plot(
    epochs,
    macro_recall_values,
    marker="o",
    label="Macro Recall"
)

plt.xlabel(
    "Epoch"
)

plt.ylabel(
    "Score"
)

plt.title(
    "Precision and Recall vs Epoch"
)

plt.xticks(
    epochs
)

plt.grid(
    True
)

plt.legend()

plt.tight_layout()

precision_recall_plot_path = (
    MODEL_DIR /
    "precision_recall_vs_epoch.png"
)

plt.savefig(
    precision_recall_plot_path,
    dpi=300
)

plt.show()

plt.close()


# ============================================================
# Training Loss Chart
# ============================================================

plt.figure(
    figsize=(10, 6)
)

plt.plot(
    epochs,
    train_losses,
    marker="o",
    label="Training Loss"
)

plt.xlabel(
    "Epoch"
)

plt.ylabel(
    "Loss"
)

plt.title(
    "Training Loss vs Epoch"
)

plt.xticks(
    epochs
)

plt.grid(
    True
)

plt.legend()

plt.tight_layout()

loss_plot_path = (
    MODEL_DIR /
    "loss_vs_epoch.png"
)

plt.savefig(
    loss_plot_path,
    dpi=300
)

plt.show()

plt.close()


# ============================================================
# Hamming Loss Chart
# ============================================================

plt.figure(
    figsize=(10, 6)
)

plt.plot(
    epochs,
    hamming_values,
    marker="o",
    label="Hamming Loss"
)

plt.xlabel(
    "Epoch"
)

plt.ylabel(
    "Hamming Loss"
)

plt.title(
    "Hamming Loss vs Epoch"
)

plt.xticks(
    epochs
)

plt.grid(
    True
)

plt.legend()

plt.tight_layout()

hamming_plot_path = (
    MODEL_DIR /
    "hamming_loss_vs_epoch.png"
)

plt.savefig(
    hamming_plot_path,
    dpi=300
)

plt.show()

plt.close()


# ============================================================
# Final Summary
# ============================================================

print("\n" + "=" * 70)
print("TRAINING COMPLETE")
print("=" * 70)

print(
    f"Best epoch:       {best_epoch}"
)

print(
    f"Best Macro F1:    {best_macro_f1:.6f}"
)

print(
    f"Best model:       {best_model_path}"
)

print(
    f"Metrics CSV:      {metrics_csv_path}"
)

print(
    f"Per-label CSV:    {per_label_csv}"
)

print(
    f"F1 plot:          {f1_plot_path}"
)

print(
    f"Precision/Recall: {precision_recall_plot_path}"
)

print(
    f"Loss plot:        {loss_plot_path}"
)

print(
    f"Hamming plot:     {hamming_plot_path}"
)

print("=" * 70)

