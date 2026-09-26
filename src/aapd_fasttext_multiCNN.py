from pathlib import Path
import csv

import numpy as np
import torch
import torch.nn as nn
import pandas as pd
import matplotlib.pyplot as plt

from tqdm import tqdm
from gensim.models.fasttext import load_facebook_vectors

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

FASTTEXT_MODEL = Path(
    "pretrained/fasttext/crawl-300d-2M-subword.bin"
)

MODEL_DIR = Path(
    "output/models/fasttext_aapd"
)

MODEL_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# Training Configuration
# ============================================================

BATCH_SIZE = 64

EPOCHS = 10

LEARNING_RATE = 1e-3

THRESHOLD = 0.25

MAX_TRAIN_BATCHES = None
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
# Load FastText
# ============================================================

print("\nLoading pretrained FastText...")

fasttext_model = load_facebook_vectors(
    str(FASTTEXT_MODEL)
)

print("FastText loaded successfully.")
print("Vector dimension:", fasttext_model.vector_size)


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
# FastText Document Encoder
# ============================================================

MAX_LENGTH = 128


def document_vector_sequence(text):
    """
    Convert a document into a sequence of pretrained FastText
    word embeddings.

    Output shape:
        [MAX_LENGTH, embedding_dim]

    FastText itself is NOT trained.
    """

    words = text.split()

    embedding_dim = fasttext_model.vector_size

    sequence = np.zeros(
        (MAX_LENGTH, embedding_dim),
        dtype=np.float32
    )

    for i, word in enumerate(words[:MAX_LENGTH]):

        sequence[i] = fasttext_model.get_vector(word)

    return sequence


# ============================================================
# Prepare Dataset
# ============================================================

def prepare_dataset(data):

    X = []
    Y = []

    print(
        "\nCreating FastText document sequences..."
    )

    for row in tqdm(data):

        sequence = document_vector_sequence(
            row["text"]
        )

        target = np.zeros(
            NUM_LABELS,
            dtype=np.float32
        )

        for label in row["labels"]:

            if label in label_to_id:

                target[
                    label_to_id[label]
                ] = 1.0

        X.append(sequence)
        Y.append(target)

    X = np.asarray(
        X,
        dtype=np.float32
    )

    Y = np.asarray(
        Y,
        dtype=np.float32
    )

    return X, Y


X_train, y_train = prepare_dataset(
    train_data
)

X_test, y_test = prepare_dataset(
    test_data
)


print(
    "\nTrain X shape:",
    X_train.shape
)

print(
    "Train y shape:",
    y_train.shape
)

print(
    "Test X shape:",
    X_test.shape
)

print(
    "Test y shape:",
    y_test.shape
)


# ============================================================
# Convert to Torch
# ============================================================

X_train = torch.tensor(
    X_train,
    dtype=torch.float32
)

y_train = torch.tensor(
    y_train,
    dtype=torch.float32
)

X_test = torch.tensor(
    X_test,
    dtype=torch.float32
)

y_test = torch.tensor(
    y_test,
    dtype=torch.float32
)


# ============================================================
# DataLoader
# ============================================================

train_dataset = torch.utils.data.TensorDataset(
    X_train,
    y_train
)

test_dataset = torch.utils.data.TensorDataset(
    X_test,
    y_test
)

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


# ============================================================
# Multi-Kernel Text CNN Classifier
# ============================================================

class FastTextCNNClassifier(nn.Module):

    def __init__(
        self,
        embedding_dim,
        num_labels,
        num_filters=128,
        kernel_sizes=(3, 4, 5),
        dropout=0.3
    ):

        super().__init__()

        self.kernel_sizes = kernel_sizes

        self.convs = nn.ModuleList(
            [
                nn.Conv2d(
                    in_channels=1,
                    out_channels=num_filters,
                    kernel_size=(kernel_size, embedding_dim)
                )
                for kernel_size in kernel_sizes
            ]
        )

        self.dropout = nn.Dropout(
            dropout
        )

        self.classifier = nn.Sequential(

            nn.Linear(
                num_filters * len(kernel_sizes),
                256
            ),

            nn.ReLU(),

            nn.Dropout(
                dropout
            ),

            nn.Linear(
                256,
                num_labels
            )
        )

    def forward(self, x):
        """
        Input:
            x = [batch, sequence_length, embedding_dim]

        Example:
            [64, 128, 300]

        After unsqueeze:
            [64, 1, 128, 300]

        Each convolution uses:
            3 x 300
            4 x 300
            5 x 300
        """

        x = x.unsqueeze(1)

        pooled_outputs = []

        for conv in self.convs:

            # [batch, filters, sequence_length-k+1, 1]
            feature = torch.relu(
                conv(x)
            )

            # Remove embedding-width dimension.
            # [batch, filters, sequence_length-k+1]
            feature = feature.squeeze(3)

            # Global max pooling over sequence.
            # [batch, filters]
            pooled = torch.max(
                feature,
                dim=2
            ).values

            pooled_outputs.append(
                pooled
            )

        # [batch, filters * number_of_kernels]
        x = torch.cat(
            pooled_outputs,
            dim=1
        )

        x = self.dropout(x)

        return self.classifier(x)


# ============================================================
# Model
# ============================================================

EMBEDDING_DIM = fasttext_model.vector_size

model = FastTextCNNClassifier(
    embedding_dim=EMBEDDING_DIM,
    num_labels=NUM_LABELS,
    num_filters=128,
    kernel_sizes=(3, 4, 5),
    dropout=0.3
).to(DEVICE)


print("\nModel:")
print(model)

print("\nCNN Configuration:")
print("MAX_LENGTH:", MAX_LENGTH)
print("FastText embedding dimension:", EMBEDDING_DIM)
print("CNN kernels:", "(3x300), (4x300), (5x300)")
print("CNN filters per kernel:", 128)
print("CNN dropout:", 0.3)


# ============================================================
# Loss
# ============================================================

criterion = nn.BCEWithLogitsLoss()


# ============================================================
# Optimizer
# ============================================================

optimizer = torch.optim.Adam(
    model.parameters(),
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

        for X, y in test_loader:

            X = X.to(DEVICE)
            y = y.to(DEVICE)

            logits = model(X)

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

    return metrics, y_true, y_pred


def calculate_per_label_f1(y_true, y_pred):

    per_label_f1 = f1_score(
        y_true,
        y_pred,
        average=None,
        zero_division=0
    )

    results = []

    for label_id, score in enumerate(per_label_f1):

        results.append(
            {
                "label": id_to_label[label_id],
                "f1": float(score)
            }
        )

    results.sort(
        key=lambda x: x["f1"],
        reverse=True
    )

    return results


def save_metric_history(history):

    history_df = pd.DataFrame(history)

    history_path = MODEL_DIR / "metrics_history.csv"

    history_df.to_csv(
        history_path,
        index=False
    )

    print(
        f"Metrics history saved: {history_path}"
    )

    # --------------------------------------------------------
    # Metric charts
    # --------------------------------------------------------

    metric_groups = [
        (
            "F1 vs Epoch",
            ["micro_f1", "macro_f1", "weighted_f1"]
        ),
        (
            "Precision vs Epoch",
            ["micro_precision", "macro_precision"]
        ),
        (
            "Recall vs Epoch",
            ["micro_recall", "macro_recall"]
        ),
    ]

    epochs = history_df["epoch"].tolist()

    for title, metric_names in metric_groups:

        plt.figure(figsize=(9, 5))

        for metric_name in metric_names:

            plt.plot(
                epochs,
                history_df[metric_name],
                marker="o",
                label=metric_name
            )

        plt.xlabel("Epoch")
        plt.ylabel("Score")
        plt.title(title)
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()

        filename = (
            title.lower()
            .replace(" ", "_")
            .replace("vs_", "vs_")
            + ".png"
        )

        chart_path = MODEL_DIR / filename

        plt.savefig(
            chart_path,
            dpi=150
        )

        plt.close()

        print(
            f"Chart saved: {chart_path}"
        )

    # --------------------------------------------------------
    # Loss vs Epoch
    # --------------------------------------------------------

    plt.figure(figsize=(9, 5))

    plt.plot(
        epochs,
        history_df["train_loss"],
        marker="o",
        label="train_loss"
    )

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss vs Epoch")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    loss_chart_path = MODEL_DIR / "training_loss_vs_epoch.png"

    plt.savefig(
        loss_chart_path,
        dpi=150
    )

    plt.close()

    print(
        f"Chart saved: {loss_chart_path}"
    )


# ============================================================
# Training
# ============================================================

print("\n" + "=" * 70)
print("START TRAINING")
print("=" * 70)

# Best model is selected using Macro F1.
# Change this to "micro_f1" or "weighted_f1" if desired.
BEST_METRIC = "macro_f1"

best_metric = -float("inf")
best_epoch = None
best_model_path = MODEL_DIR / "best_model.pt"

history = []

for epoch in range(1, EPOCHS + 1):

    model.train()

    running_loss = 0.0

    progress = tqdm(
        enumerate(train_loader),
        total=len(train_loader),
        desc=f"Epoch {epoch}/{EPOCHS}"
    )

    for batch_idx, (X, y) in progress:

        if (
            MAX_TRAIN_BATCHES is not None
            and batch_idx >= MAX_TRAIN_BATCHES
        ):
            break

        X = X.to(DEVICE)
        y = y.to(DEVICE)

        optimizer.zero_grad()

        logits = model(X)

        loss = criterion(
            logits,
            y
        )

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

    # --------------------------------------------------------
    # Evaluation after every epoch
    # --------------------------------------------------------

    metrics, _, _ = evaluate()

    epoch_results = {
        "epoch": epoch,
        "train_loss": avg_loss,
        **metrics
    }

    history.append(epoch_results)

    print("\n" + "-" * 70)
    print(f"Epoch {epoch}/{EPOCHS}")
    print(f"Train Loss:       {avg_loss:.6f}")
    print(f"Micro F1:         {metrics['micro_f1']:.4f}")
    print(f"Macro F1:         {metrics['macro_f1']:.4f}")
    print(f"Weighted F1:      {metrics['weighted_f1']:.4f}")
    print(f"Micro Precision:  {metrics['micro_precision']:.4f}")
    print(f"Macro Precision:  {metrics['macro_precision']:.4f}")
    print(f"Micro Recall:     {metrics['micro_recall']:.4f}")
    print(f"Macro Recall:     {metrics['macro_recall']:.4f}")
    print(f"Hamming Loss:     {metrics['hamming_loss']:.6f}")

    # --------------------------------------------------------
    # Save metrics after every epoch
    # --------------------------------------------------------

    metrics_path = MODEL_DIR / "metrics_history.csv"

    pd.DataFrame(history).to_csv(
        metrics_path,
        index=False
    )

    # --------------------------------------------------------
    # Save best model
    # --------------------------------------------------------

    current_metric = metrics[BEST_METRIC]

    if current_metric > best_metric:

        best_metric = current_metric
        best_epoch = epoch

        torch.save(
            {
                "epoch": epoch,
                "best_metric": BEST_METRIC,
                "best_metric_value": best_metric,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "label_to_id": label_to_id,
                "embedding_dim": EMBEDDING_DIM,
                "num_labels": NUM_LABELS,
                "threshold": THRESHOLD,
                "max_length": MAX_LENGTH,
                "num_filters": 128,
                "kernel_sizes": (3, 4, 5),
                "dropout": 0.3,
            },
            best_model_path
        )

        print(
            f"*** New best model saved: "
            f"{BEST_METRIC}={best_metric:.6f}"
        )

    else:

        print(
            f"Best {BEST_METRIC}: "
            f"{best_metric:.6f} "
            f"(epoch {best_epoch})"
        )


# ============================================================
# Load Best Model
# ============================================================

print("\n" + "=" * 70)
print("LOADING BEST MODEL")
print("=" * 70)

best_checkpoint = torch.load(
    best_model_path,
    map_location=DEVICE,
    weights_only=False
)

model.load_state_dict(
    best_checkpoint["model_state_dict"]
)

print(
    f"Best model epoch: {best_checkpoint['epoch']}"
)

print(
    f"Best {BEST_METRIC}: "
    f"{best_checkpoint['best_metric_value']:.6f}"
)


# ============================================================
# Final Evaluation
# ============================================================

print("\n" + "=" * 70)
print("FINAL RESULTS - BEST MODEL")
print("=" * 70)

final_metrics, y_true, y_pred = evaluate()

for name, value in final_metrics.items():

    print(
        f"{name:20s}: {value:.6f}"
    )


# ============================================================
# Per-label F1
# ============================================================

print("\n" + "=" * 70)
print("PER-LABEL F1 - SORTED DESCENDING")
print("=" * 70)

per_label_results = calculate_per_label_f1(
    y_true,
    y_pred
)

per_label_df = pd.DataFrame(
    per_label_results
)

for rank, row in enumerate(
    per_label_results,
    start=1
):

    print(
        f"{rank:3d}. "
        f"{row['label']:15s} "
        f"F1: {row['f1']:.6f}"
    )

per_label_path = (
    MODEL_DIR / "per_label_f1.csv"
)

per_label_df.to_csv(
    per_label_path,
    index=False
)

print(
    f"\nPer-label F1 saved: {per_label_path}"
)


# ============================================================
# Save Final Metrics
# ============================================================

final_metrics_path = (
    MODEL_DIR / "final_metrics.csv"
)

pd.DataFrame(
    [
        {
            "best_epoch": best_epoch,
            "best_metric": BEST_METRIC,
            "best_metric_value": best_metric,
            **final_metrics
        }
    ]
).to_csv(
    final_metrics_path,
    index=False
)

print(
    f"Final metrics saved: {final_metrics_path}"
)


# ============================================================
# Save Charts
# ============================================================

save_metric_history(history)

print("\n" + "=" * 70)
print("TRAINING COMPLETE")
print("=" * 70)
print(f"Best model:      {best_model_path}")
print(f"Best epoch:      {best_epoch}")
print(f"Best {BEST_METRIC}: {best_metric:.6f}")
print(f"Metrics CSV:     {MODEL_DIR / 'metrics_history.csv'}")
print(f"Per-label F1:    {per_label_path}")
print("=" * 70)
