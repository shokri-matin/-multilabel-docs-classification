from pathlib import Path

import csv
import json
import shutil

import matplotlib.pyplot as plt
import torch
import torch.nn as nn

from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel

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

MODEL_NAME = "bert-base-uncased"

# ------------------------------------------------------------
# Training configuration
# ------------------------------------------------------------

BATCH_SIZE = 16
EPOCHS = 10

# Maximum TOTAL training batches.
#
# None -> train all batches in all epochs.
# 5    -> useful for a quick test, but only the first 5 batches
#         are trained in total, so later epochs will not train.
MAX_TRAIN_BATCHES = 5

LEARNING_RATE = 2e-5

# ------------------------------------------------------------
# Multi-label prediction threshold
# ------------------------------------------------------------

THRESHOLD = 0.5

# ------------------------------------------------------------
# Maximum BERT sequence length
# ------------------------------------------------------------

MAX_LENGTH = 32

# ------------------------------------------------------------
# Best-model selection
#
# NOTE:
# The current script has only train/test CSV files, so TEST_CSV
# is evaluated after every epoch and is used for model selection.
# For a rigorous final experiment, use a separate validation CSV
# for BEST_METRIC selection and keep TEST_CSV for final evaluation.
# ------------------------------------------------------------

BEST_METRIC = "macro_f1"
BEST_MODE = "max"

# ------------------------------------------------------------
# Output
# ------------------------------------------------------------

CHECKPOINT_DIR = Path(
    "output/models/bert_aapd"
)

METRICS_DIR = CHECKPOINT_DIR / "metrics"
PLOTS_DIR = CHECKPOINT_DIR / "plots"

CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
METRICS_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# Device
# ============================================================

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ============================================================
# Discover AAPD labels
# ============================================================

def discover_labels(csv_file):
    """
    Discover all labels from the training CSV.

    Expected CSV schema:
        docid,sec,label

    Example:
        2286,some document text,E11|ECAT
    """

    labels = set()

    with csv_file.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as f:

        reader = csv.DictReader(f)

        required_columns = {
            "docid",
            "sec",
            "label",
        }

        missing_columns = (
            required_columns
            - set(reader.fieldnames or [])
        )

        if missing_columns:
            raise ValueError(
                f"Missing CSV columns: "
                f"{sorted(missing_columns)}"
            )

        for row in reader:

            label_string = (
                row["label"] or ""
            )

            for label in label_string.split("|"):

                label = label.strip()

                if label:
                    labels.add(label)

    return sorted(labels)


# ============================================================
# Count CSV rows
# ============================================================

def count_csv_rows(csv_file):
    """Count the number of data rows in a CSV file."""

    with csv_file.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as f:

        reader = csv.DictReader(f)

        return sum(
            1
            for _ in reader
        )


# ============================================================
# Count CSV batches
# ============================================================

def count_batches(
    csv_file,
    batch_size,
):
    """Count the total number of batches in a CSV file."""

    row_count = count_csv_rows(
        csv_file
    )

    return (
        row_count
        + batch_size
        - 1
    ) // batch_size


# ============================================================
# Streaming CSV reader
# ============================================================

def read_batches(
    csv_file,
    batch_size,
):
    """
    Read CSV in batches.

    Expected CSV schema:
        docid,sec,label
    """

    texts = []
    labels = []

    with csv_file.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as f:

        reader = csv.DictReader(f)

        required_columns = {
            "docid",
            "sec",
            "label",
        }

        missing_columns = (
            required_columns
            - set(reader.fieldnames or [])
        )

        if missing_columns:
            raise ValueError(
                f"Missing CSV columns: "
                f"{sorted(missing_columns)}"
            )

        for row in reader:

            texts.append(
                row["sec"]
            )

            labels.append(
                row["label"]
            )

            if len(texts) == batch_size:

                yield texts, labels

                texts = []
                labels = []

        # Last incomplete batch
        if texts:
            yield texts, labels


# ============================================================
# Convert labels to multi-hot
# ============================================================

def encode_labels(
    labels,
    label_to_id,
    num_labels,
):
    """Convert pipe-separated labels into multi-hot vectors."""

    target = torch.zeros(
        len(labels),
        num_labels,
        dtype=torch.float32,
    )

    for row_idx, label_string in enumerate(labels):

        for label in label_string.split("|"):

            label = label.strip()

            if not label:
                continue

            if label not in label_to_id:
                raise ValueError(
                    f"Unknown label: {label}"
                )

            label_id = label_to_id[label]

            target[
                row_idx,
                label_id
            ] = 1.0

    return target


# ============================================================
# BERT + Multi-Label Classifier
# ============================================================

class BertMultiLabelClassifier(nn.Module):

    def __init__(
        self,
        model_name,
        num_labels,
    ):

        super().__init__()

        # Load pretrained BERT
        self.bert = AutoModel.from_pretrained(
            model_name
        )

        # Freeze BERT
        for param in self.bert.parameters():
            param.requires_grad = False

        hidden_size = self.bert.config.hidden_size

        # Two-layer trainable classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, num_labels),
        )

    def forward(
        self,
        input_ids,
        attention_mask,
    ):

        # BERT is frozen
        with torch.no_grad():

            outputs = self.bert(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )

        # CLS embedding
        cls_embedding = (
            outputs.last_hidden_state[:, 0]
        )

        # Classifier
        logits = self.classifier(
            cls_embedding
        )

        return logits


# ============================================================
# Save checkpoint
# ============================================================

def save_checkpoint(
    model,
    tokenizer,
    optimizer,
    epoch,
    total_batches,
    average_loss,
    labels,
    checkpoint_dir,
    extra_state=None,
):
    """
    Save:
        1. Frozen pretrained BERT
        2. Tokenizer
        3. Trainable classifier
        4. Optimizer state
        5. Training configuration
        6. AAPD label vocabulary
        7. Optional best-model metadata
    """

    checkpoint_dir = Path(checkpoint_dir)

    checkpoint_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print()
    print("Saving checkpoint to:")
    print(checkpoint_dir)

    # --------------------------------------------------------
    # Save BERT
    # --------------------------------------------------------

    model.bert.save_pretrained(
        checkpoint_dir
    )

    # --------------------------------------------------------
    # Save tokenizer
    # --------------------------------------------------------

    tokenizer.save_pretrained(
        checkpoint_dir
    )

    # --------------------------------------------------------
    # Save classifier
    # --------------------------------------------------------

    torch.save(
        model.classifier.state_dict(),
        checkpoint_dir / "classifier.pt",
    )

    # --------------------------------------------------------
    # Save complete classifier + optimizer checkpoint
    # --------------------------------------------------------

    training_state = {
        "epoch": epoch,
        "total_batches": total_batches,
        "average_loss": average_loss,
        "learning_rate": LEARNING_RATE,
        "batch_size": BATCH_SIZE,
        "max_length": MAX_LENGTH,
        "max_train_batches": MAX_TRAIN_BATCHES,
        "threshold": THRESHOLD,
        "num_labels": len(labels),
        "labels": labels,
        "bert_frozen": True,
        "optimizer_state_dict": optimizer.state_dict(),
    }

    if extra_state is not None:
        training_state.update(extra_state)

    torch.save(
        training_state,
        checkpoint_dir / "training_state.pt",
    )

    print("Checkpoint saved successfully.")

    return checkpoint_dir


# ============================================================
# Evaluation
# ============================================================

def evaluate(
    model,
    tokenizer,
    test_csv,
    label_to_id,
    num_labels,
    labels,
    show_header=True,
):
    """
    Evaluate the classifier.

    Returns:
        metrics:
            Aggregate metrics.
        per_label_f1:
            Dictionary mapping label -> F1.
    """

    if show_header:
        print()
        print("=" * 70)
        print("AAPD TEST EVALUATION")
        print("=" * 70)

    model.eval()

    all_predictions = []
    all_targets = []

    criterion = nn.BCEWithLogitsLoss()

    total_loss = 0.0
    batch_count = 0

    batches = read_batches(
        test_csv,
        BATCH_SIZE,
    )

    total_test_batches = count_batches(
        test_csv,
        BATCH_SIZE,
    )

    progress = tqdm(
        batches,
        total=total_test_batches,
        desc="Evaluating",
        unit="batch",
        dynamic_ncols=True,
    )

    with torch.no_grad():

        for texts, label_strings in progress:

            # Tokenization
            encoded = tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=MAX_LENGTH,
                return_tensors="pt",
            )

            input_ids = (
                encoded["input_ids"]
                .to(DEVICE)
            )

            attention_mask = (
                encoded["attention_mask"]
                .to(DEVICE)
            )

            # Targets
            targets = encode_labels(
                label_strings,
                label_to_id,
                num_labels,
            ).to(DEVICE)

            # Forward
            logits = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )

            # Loss
            loss = criterion(
                logits,
                targets,
            )

            total_loss += loss.item()
            batch_count += 1

            # Predictions
            probabilities = torch.sigmoid(
                logits
            )

            predictions = (
                probabilities >= THRESHOLD
            ).float()

            all_predictions.append(
                predictions.cpu()
            )

            all_targets.append(
                targets.cpu()
            )

            progress.set_postfix(
                done=batch_count,
                remaining=(
                    total_test_batches
                    - batch_count
                ),
            )

    y_pred = torch.cat(
        all_predictions,
        dim=0,
    ).numpy()

    y_true = torch.cat(
        all_targets,
        dim=0,
    ).numpy()

    average_test_loss = (
        total_loss
        / max(1, batch_count)
    )

    # --------------------------------------------------------
    # Aggregate metrics
    # --------------------------------------------------------

    micro_f1 = f1_score(
        y_true,
        y_pred,
        average="micro",
        zero_division=0,
    )

    macro_f1 = f1_score(
        y_true,
        y_pred,
        average="macro",
        zero_division=0,
    )

    weighted_f1 = f1_score(
        y_true,
        y_pred,
        average="weighted",
        zero_division=0,
    )

    micro_precision = precision_score(
        y_true,
        y_pred,
        average="micro",
        zero_division=0,
    )

    macro_precision = precision_score(
        y_true,
        y_pred,
        average="macro",
        zero_division=0,
    )

    weighted_precision = precision_score(
        y_true,
        y_pred,
        average="weighted",
        zero_division=0,
    )

    micro_recall = recall_score(
        y_true,
        y_pred,
        average="micro",
        zero_division=0,
    )

    macro_recall = recall_score(
        y_true,
        y_pred,
        average="macro",
        zero_division=0,
    )

    weighted_recall = recall_score(
        y_true,
        y_pred,
        average="weighted",
        zero_division=0,
    )

    hamming = hamming_loss(
        y_true,
        y_pred,
    )

    # --------------------------------------------------------
    # Per-label F1
    # --------------------------------------------------------

    per_label_f1_values = f1_score(
        y_true,
        y_pred,
        average=None,
        zero_division=0,
    )

    per_label_f1 = {
        label: float(score)
        for label, score in zip(
            labels,
            per_label_f1_values,
        )
    }

    metrics = {
        "test_loss": float(average_test_loss),

        "micro_f1": float(micro_f1),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),

        "micro_precision": float(micro_precision),
        "macro_precision": float(macro_precision),
        "weighted_precision": float(weighted_precision),

        "micro_recall": float(micro_recall),
        "macro_recall": float(macro_recall),
        "weighted_recall": float(weighted_recall),

        "hamming_loss": float(hamming),
    }

    # --------------------------------------------------------
    # Print results
    # --------------------------------------------------------

    print()
    print("-" * 70)
    print("TEST RESULTS")
    print("-" * 70)

    print(f"Test Loss       : {average_test_loss:.6f}")
    print(f"Micro F1        : {micro_f1:.6f}")
    print(f"Macro F1        : {macro_f1:.6f}")
    print(f"Weighted F1     : {weighted_f1:.6f}")

    print(f"Micro Precision : {micro_precision:.6f}")
    print(f"Macro Precision : {macro_precision:.6f}")
    print(f"Weighted Prec.  : {weighted_precision:.6f}")

    print(f"Micro Recall    : {micro_recall:.6f}")
    print(f"Macro Recall    : {macro_recall:.6f}")
    print(f"Weighted Recall : {weighted_recall:.6f}")

    print(f"Hamming Loss    : {hamming:.6f}")
    print(f"Threshold       : {THRESHOLD}")
    print(f"Test samples    : {len(y_true):,}")
    print(f"Number of labels: {num_labels}")

    print("-" * 70)

    return metrics, per_label_f1


# ============================================================
# Save epoch metrics
# ============================================================

def save_epoch_metrics(
    history,
):
    """
    Save aggregate metrics for every epoch to CSV and JSON.
    """

    if not history:
        return

    csv_file = METRICS_DIR / "epoch_metrics.csv"
    json_file = METRICS_DIR / "epoch_metrics.json"

    fieldnames = list(history[0].keys())

    with csv_file.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(history)

    with json_file.open(
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            history,
            f,
            indent=2,
        )

    print()
    print("Epoch metrics saved to:")
    print(csv_file)
    print(json_file)


# ============================================================
# Save per-label F1
# ============================================================

def save_per_label_f1(
    per_label_f1,
    epoch=None,
):
    """
    Save per-label F1 sorted from highest to lowest.
    """

    if epoch is None:
        filename = "per_label_f1_final.csv"
    else:
        filename = (
            f"per_label_f1_epoch_{epoch}.csv"
        )

    output_file = METRICS_DIR / filename

    sorted_items = sorted(
        per_label_f1.items(),
        key=lambda item: item[1],
        reverse=True,
    )

    with output_file.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as f:

        writer = csv.writer(f)

        writer.writerow([
            "rank",
            "label",
            "f1",
        ])

        for rank, (label, score) in enumerate(
            sorted_items,
            start=1,
        ):

            writer.writerow([
                rank,
                label,
                f"{score:.6f}",
            ])

    return output_file


# ============================================================
# Print per-label F1
# ============================================================

def print_per_label_f1(
    per_label_f1,
):
    """Print per-label F1 sorted descending."""

    sorted_items = sorted(
        per_label_f1.items(),
        key=lambda item: item[1],
        reverse=True,
    )

    print()
    print("=" * 70)
    print("FINAL PER-LABEL F1 — SORTED DESCENDING")
    print("=" * 70)

    print(
        f"{'Rank':<8}"
        f"{'Label':<20}"
        f"{'F1':>10}"
    )

    print("-" * 70)

    for rank, (label, score) in enumerate(
        sorted_items,
        start=1,
    ):

        print(
            f"{rank:<8}"
            f"{label:<20}"
            f"{score:>10.6f}"
        )

    print("=" * 70)


# ============================================================
# Plot training/evaluation metrics
# ============================================================

def plot_training_metrics(
    history,
):
    """
    Save charts showing metrics versus epoch.

    Charts:
        1. Loss
        2. F1
        3. Precision
        4. Recall
        5. Hamming loss
    """

    if not history:
        return

    epochs = [
        row["epoch"]
        for row in history
    ]

    # --------------------------------------------------------
    # Loss
    # --------------------------------------------------------

    plt.figure(figsize=(10, 6))

    plt.plot(
        epochs,
        [
            row["train_loss"]
            for row in history
        ],
        marker="o",
        label="Train Loss",
    )

    plt.plot(
        epochs,
        [
            row["test_loss"]
            for row in history
        ],
        marker="o",
        label="Test Loss",
    )

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss vs Epoch")
    plt.xticks(epochs)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        PLOTS_DIR / "loss_vs_epoch.png",
        dpi=200,
    )

    plt.close()

    # --------------------------------------------------------
    # F1
    # --------------------------------------------------------

    plt.figure(figsize=(10, 6))

    plt.plot(
        epochs,
        [
            row["micro_f1"]
            for row in history
        ],
        marker="o",
        label="Micro F1",
    )

    plt.plot(
        epochs,
        [
            row["macro_f1"]
            for row in history
        ],
        marker="o",
        label="Macro F1",
    )

    plt.plot(
        epochs,
        [
            row["weighted_f1"]
            for row in history
        ],
        marker="o",
        label="Weighted F1",
    )

    plt.xlabel("Epoch")
    plt.ylabel("F1")
    plt.title("F1 vs Epoch")
    plt.xticks(epochs)
    plt.ylim(0, 1)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        PLOTS_DIR / "f1_vs_epoch.png",
        dpi=200,
    )

    plt.close()

    # --------------------------------------------------------
    # Precision
    # --------------------------------------------------------

    plt.figure(figsize=(10, 6))

    plt.plot(
        epochs,
        [
            row["micro_precision"]
            for row in history
        ],
        marker="o",
        label="Micro Precision",
    )

    plt.plot(
        epochs,
        [
            row["macro_precision"]
            for row in history
        ],
        marker="o",
        label="Macro Precision",
    )

    plt.plot(
        epochs,
        [
            row["weighted_precision"]
            for row in history
        ],
        marker="o",
        label="Weighted Precision",
    )

    plt.xlabel("Epoch")
    plt.ylabel("Precision")
    plt.title("Precision vs Epoch")
    plt.xticks(epochs)
    plt.ylim(0, 1)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        PLOTS_DIR / "precision_vs_epoch.png",
        dpi=200,
    )

    plt.close()

    # --------------------------------------------------------
    # Recall
    # --------------------------------------------------------

    plt.figure(figsize=(10, 6))

    plt.plot(
        epochs,
        [
            row["micro_recall"]
            for row in history
        ],
        marker="o",
        label="Micro Recall",
    )

    plt.plot(
        epochs,
        [
            row["macro_recall"]
            for row in history
        ],
        marker="o",
        label="Macro Recall",
    )

    plt.plot(
        epochs,
        [
            row["weighted_recall"]
            for row in history
        ],
        marker="o",
        label="Weighted Recall",
    )

    plt.xlabel("Epoch")
    plt.ylabel("Recall")
    plt.title("Recall vs Epoch")
    plt.xticks(epochs)
    plt.ylim(0, 1)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        PLOTS_DIR / "recall_vs_epoch.png",
        dpi=200,
    )

    plt.close()

    # --------------------------------------------------------
    # Hamming loss
    # --------------------------------------------------------

    plt.figure(figsize=(10, 6))

    plt.plot(
        epochs,
        [
            row["hamming_loss"]
            for row in history
        ],
        marker="o",
        label="Hamming Loss",
    )

    plt.xlabel("Epoch")
    plt.ylabel("Hamming Loss")
    plt.title("Hamming Loss vs Epoch")
    plt.xticks(epochs)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        PLOTS_DIR / "hamming_loss_vs_epoch.png",
        dpi=200,
    )

    plt.close()

    print()
    print("Metric charts saved to:")
    print(PLOTS_DIR)


# ============================================================
# Save final evaluation results
# ============================================================

def save_evaluation_results(
    checkpoint_dir,
    metrics,
):
    """Save final aggregate evaluation metrics."""

    checkpoint_dir = Path(checkpoint_dir)

    evaluation_file = (
        checkpoint_dir
        / "evaluation_results.txt"
    )

    with evaluation_file.open(
        "w",
        encoding="utf-8",
    ) as f:

        f.write(
            "AAPD BERT Evaluation Results\n"
        )

        f.write("=" * 60 + "\n")

        for name, value in metrics.items():

            f.write(
                f"{name}: "
                f"{value:.6f}\n"
            )

        f.write(
            f"threshold: "
            f"{THRESHOLD}\n"
        )

        f.write(
            f"batch_size: "
            f"{BATCH_SIZE}\n"
        )

        f.write(
            f"max_length: "
            f"{MAX_LENGTH}\n"
        )

        f.write(
            f"max_train_batches: "
            f"{MAX_TRAIN_BATCHES}\n"
        )

    print()
    print("Evaluation results saved to:")
    print(evaluation_file)


# ============================================================
# Main
# ============================================================

def main():

    print("=" * 70)
    print(
        "AAPD BERT Feature Extraction "
        "+ Multi-Label Classifier"
    )
    print("=" * 70)

    print(f"Device: {DEVICE}")
    print(f"Train CSV: {TRAIN_CSV}")
    print(f"Test CSV: {TEST_CSV}")
    print(f"Checkpoint directory: {CHECKPOINT_DIR}")

    print()
    print("BERT training: DISABLED")
    print("Classifier training: ENABLED")

    print()
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Epochs: {EPOCHS}")
    print(
        f"Maximum training batches: "
        f"{MAX_TRAIN_BATCHES}"
    )
    print(f"Learning rate: {LEARNING_RATE}")
    print(f"Threshold: {THRESHOLD}")
    print(f"Best-model metric: {BEST_METRIC}")

    # ========================================================
    # Validate files
    # ========================================================

    if not TRAIN_CSV.exists():
        raise FileNotFoundError(
            f"Training CSV not found:\n"
            f"{TRAIN_CSV}"
        )

    if not TEST_CSV.exists():
        raise FileNotFoundError(
            f"Test CSV not found:\n"
            f"{TEST_CSV}"
        )

    # ========================================================
    # Calculate training batches
    # ========================================================

    print()
    print("Calculating training batches...")

    total_dataset_batches = count_batches(
        TRAIN_CSV,
        BATCH_SIZE,
    )

    if MAX_TRAIN_BATCHES is None:

        total_training_batches = (
            total_dataset_batches
            * EPOCHS
        )

    else:

        total_training_batches = min(
            MAX_TRAIN_BATCHES,
            total_dataset_batches * EPOCHS,
        )

    print(
        f"Dataset batches per epoch : "
        f"{total_dataset_batches:,}"
    )

    print(
        f"Total possible batches     : "
        f"{total_dataset_batches * EPOCHS:,}"
    )

    print(
        f"Total training batches     : "
        f"{total_training_batches:,}"
    )

    # ========================================================
    # Discover labels
    # ========================================================

    print()
    print("Discovering AAPD labels...")

    labels = discover_labels(
        TRAIN_CSV
    )

    num_labels = len(labels)

    label_to_id = {
        label: idx
        for idx, label in enumerate(labels)
    }

    print(
        f"Number of labels: "
        f"{num_labels}"
    )

    print()
    print("Labels:")
    print(labels)

    # ========================================================
    # Tokenizer
    # ========================================================

    print()
    print("Loading tokenizer...")

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME
    )

    # ========================================================
    # Model
    # ========================================================

    print("Loading BERT model...")

    model = BertMultiLabelClassifier(
        MODEL_NAME,
        num_labels,
    )

    model.to(DEVICE)

    # ========================================================
    # Verify BERT frozen
    # ========================================================

    bert_trainable_params = sum(
        p.numel()
        for p in model.bert.parameters()
        if p.requires_grad
    )

    classifier_trainable_params = sum(
        p.numel()
        for p in model.classifier.parameters()
        if p.requires_grad
    )

    print()
    print("Trainable parameter check:")

    print(
        f"BERT trainable parameters: "
        f"{bert_trainable_params:,}"
    )

    print(
        f"Classifier trainable parameters: "
        f"{classifier_trainable_params:,}"
    )

    if bert_trainable_params != 0:
        raise RuntimeError(
            "BERT is not completely frozen!"
        )

    if classifier_trainable_params == 0:
        raise RuntimeError(
            "Classifier has no trainable parameters!"
        )

    # ========================================================
    # Optimizer
    # ========================================================

    optimizer = torch.optim.AdamW(
        model.classifier.parameters(),
        lr=LEARNING_RATE,
    )

    # ========================================================
    # Loss
    # ========================================================

    criterion = nn.BCEWithLogitsLoss()

    # ========================================================
    # Training state
    # ========================================================

    global_batch_count = 0
    last_epoch = 0
    last_average_loss = 0.0

    history = []

    best_metric_value = (
        float("-inf")
        if BEST_MODE == "max"
        else float("inf")
    )

    best_epoch = None
    best_checkpoint_dir = (
        CHECKPOINT_DIR / "best_model"
    )

    # Remove previous best checkpoint so an old model cannot
    # accidentally be reported as the new best model.
    if best_checkpoint_dir.exists():
        shutil.rmtree(best_checkpoint_dir)

    # ========================================================
    # Training
    # ========================================================

    for epoch in range(EPOCHS):

        last_epoch = epoch + 1

        # If the global training limit has already been reached,
        # stop instead of creating empty epochs.
        if (
            MAX_TRAIN_BATCHES is not None
            and global_batch_count >= MAX_TRAIN_BATCHES
        ):
            break

        model.train()

        # BERT explicitly stays in evaluation mode
        model.bert.eval()

        total_loss = 0.0
        epoch_batch_count = 0

        print()
        print("=" * 70)
        print(
            f"Epoch {epoch + 1}/{EPOCHS}"
        )
        print(
            f"Global batches completed: "
            f"{global_batch_count}"
        )
        print("=" * 70)

        batches = read_batches(
            TRAIN_CSV,
            BATCH_SIZE,
        )

        progress = tqdm(
            batches,
            total=total_dataset_batches,
            desc=(
                f"Epoch {epoch + 1}/{EPOCHS}"
            ),
            unit="batch",
            dynamic_ncols=True,
        )

        # ====================================================
        # Batch loop
        # ====================================================

        for texts, label_strings in progress:

            # ------------------------------------------------
            # Check maximum number of batches
            # ------------------------------------------------

            if (
                MAX_TRAIN_BATCHES is not None
                and global_batch_count
                >= MAX_TRAIN_BATCHES
            ):
                break

            # ------------------------------------------------
            # Tokenization
            # ------------------------------------------------

            encoded = tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=MAX_LENGTH,
                return_tensors="pt",
            )

            input_ids = (
                encoded["input_ids"]
                .to(DEVICE)
            )

            attention_mask = (
                encoded["attention_mask"]
                .to(DEVICE)
            )

            # ------------------------------------------------
            # Labels
            # ------------------------------------------------

            targets = encode_labels(
                label_strings,
                label_to_id,
                num_labels,
            ).to(DEVICE)

            # ------------------------------------------------
            # Forward
            # ------------------------------------------------

            optimizer.zero_grad(
                set_to_none=True
            )

            logits = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )

            # ------------------------------------------------
            # Loss
            # ------------------------------------------------

            loss = criterion(
                logits,
                targets,
            )

            # ------------------------------------------------
            # Backward
            # ------------------------------------------------

            loss.backward()

            # ------------------------------------------------
            # Update classifier
            # ------------------------------------------------

            optimizer.step()

            # ------------------------------------------------
            # Statistics
            # ------------------------------------------------

            loss_value = loss.item()

            total_loss += loss_value
            epoch_batch_count += 1
            global_batch_count += 1

            average_running_loss = (
                total_loss
                / epoch_batch_count
            )

            progress_percent = (
                global_batch_count
                / max(1, total_training_batches)
                * 100
            )

            progress.set_postfix(
                done=f"{global_batch_count:,}",
                progress=f"{progress_percent:.1f}%",
                loss=f"{loss_value:.4f}",
                avg_loss=f"{average_running_loss:.4f}",
            )

        progress.close()

        # ====================================================
        # Epoch training statistics
        # ====================================================

        average_loss = (
            total_loss
            / max(
                1,
                epoch_batch_count,
            )
        )

        last_average_loss = average_loss

        print()
        print("-" * 70)
        print(
            f"Epoch {epoch + 1} training completed"
        )
        print(
            f"Epoch batches : "
            f"{epoch_batch_count:,}"
        )
        print(
            f"Total batches : "
            f"{global_batch_count:,}"
        )
        print(
            f"Train loss    : "
            f"{average_loss:.6f}"
        )
        print("-" * 70)

        # ====================================================
        # Evaluate after EVERY epoch
        # ====================================================

        epoch_metrics, epoch_per_label_f1 = evaluate(
            model=model,
            tokenizer=tokenizer,
            test_csv=TEST_CSV,
            label_to_id=label_to_id,
            num_labels=num_labels,
            labels=labels,
            show_header=True,
        )

        # Save per-label F1 for this epoch
        epoch_per_label_file = save_per_label_f1(
            epoch_per_label_f1,
            epoch=epoch + 1,
        )

        # ====================================================
        # Record epoch history
        # ====================================================

        history_row = {
            "epoch": epoch + 1,
            "train_loss": float(average_loss),
            "train_batches": int(epoch_batch_count),
            "total_batches": int(global_batch_count),
        }

        history_row.update(epoch_metrics)

        history.append(history_row)

        # Save metrics immediately after the epoch.
        save_epoch_metrics(history)

        print()
        print(
            f"Epoch {epoch + 1} metrics saved."
        )
        print(
            f"Per-label F1 saved to: "
            f"{epoch_per_label_file}"
        )

        # ====================================================
        # Save normal epoch checkpoint
        # ====================================================

        epoch_checkpoint_dir = (
            CHECKPOINT_DIR
            / f"epoch_{epoch + 1}"
        )

        save_checkpoint(
            model=model,
            tokenizer=tokenizer,
            optimizer=optimizer,
            epoch=epoch + 1,
            total_batches=global_batch_count,
            average_loss=average_loss,
            labels=labels,
            checkpoint_dir=epoch_checkpoint_dir,
            extra_state={
                "epoch_metrics": epoch_metrics,
                "best_metric_name": BEST_METRIC,
            },
        )

        # ====================================================
        # Best-model check
        # ====================================================

        current_metric = epoch_metrics[
            BEST_METRIC
        ]

        if BEST_MODE == "max":
            is_better = (
                current_metric
                > best_metric_value
            )
        else:
            is_better = (
                current_metric
                < best_metric_value
            )

        if is_better:

            best_metric_value = current_metric
            best_epoch = epoch + 1

            # Remove previous best model.
            if best_checkpoint_dir.exists():
                shutil.rmtree(
                    best_checkpoint_dir
                )

            save_checkpoint(
                model=model,
                tokenizer=tokenizer,
                optimizer=optimizer,
                epoch=epoch + 1,
                total_batches=global_batch_count,
                average_loss=average_loss,
                labels=labels,
                checkpoint_dir=best_checkpoint_dir,
                extra_state={
                    "best_model": True,
                    "best_metric_name": BEST_METRIC,
                    "best_metric_value": float(
                        best_metric_value
                    ),
                    "best_epoch": best_epoch,
                    "epoch_metrics": epoch_metrics,
                },
            )

            print()
            print("=" * 70)
            print("NEW BEST MODEL")
            print("=" * 70)
            print(
                f"Metric : {BEST_METRIC}"
            )
            print(
                f"Value  : "
                f"{best_metric_value:.6f}"
            )
            print(
                f"Epoch  : {best_epoch}"
            )
            print(
                f"Saved  : {best_checkpoint_dir}"
            )
            print("=" * 70)

        else:

            print()
            print(
                f"Best model unchanged. "
                f"Current {BEST_METRIC}: "
                f"{current_metric:.6f}, "
                f"best: "
                f"{best_metric_value:.6f}"
            )

        # ====================================================
        # Stop after maximum batch limit
        # ====================================================

        if (
            MAX_TRAIN_BATCHES is not None
            and global_batch_count
            >= MAX_TRAIN_BATCHES
        ):
            print()
            print("=" * 70)
            print("MAXIMUM TRAINING BATCHES REACHED")
            print("=" * 70)
            print(
                f"Maximum batches: "
                f"{MAX_TRAIN_BATCHES:,}"
            )
            print(
                f"Actual batches: "
                f"{global_batch_count:,}"
            )
            print("Training stopped.")
            print("=" * 70)
            break

    # ========================================================
    # Training finished
    # ========================================================

    print()
    print("=" * 70)
    print("TRAINING COMPLETED")
    print("=" * 70)

    print(
        f"Final epoch: "
        f"{last_epoch}"
    )

    print(
        f"Total training batches: "
        f"{global_batch_count:,}"
    )

    print(
        f"Final training loss: "
        f"{last_average_loss:.6f}"
    )

    print(
        "BERT remained frozen."
    )

    print(
        "Only the classifier was trained."
    )

    # ========================================================
    # Final charts
    # ========================================================

    plot_training_metrics(
        history
    )

    # ========================================================
    # Final evaluation
    #
    # Evaluate the BEST model, not simply the last epoch.
    # ========================================================

    if best_epoch is not None:

        print()
        print("=" * 70)
        print(
            "LOADING BEST MODEL FOR FINAL EVALUATION"
        )
        print("=" * 70)

        classifier_path = (
            best_checkpoint_dir
            / "classifier.pt"
        )

        if not classifier_path.exists():
            raise FileNotFoundError(
                f"Best classifier not found:\n"
                f"{classifier_path}"
            )

        model.classifier.load_state_dict(
            torch.load(
                classifier_path,
                map_location=DEVICE,
            )
        )

        model.to(DEVICE)

    final_metrics, final_per_label_f1 = evaluate(
        model=model,
        tokenizer=tokenizer,
        test_csv=TEST_CSV,
        label_to_id=label_to_id,
        num_labels=num_labels,
        labels=labels,
        show_header=True,
    )

    # ========================================================
    # Save final results
    # ========================================================

    final_per_label_file = save_per_label_f1(
        final_per_label_f1
    )

    print_per_label_f1(
        final_per_label_f1
    )

    # Save final aggregate metrics.
    save_evaluation_results(
        checkpoint_dir=best_checkpoint_dir,
        metrics=final_metrics,
    )

    # Save final JSON summary.
    final_summary = {
        "best_epoch": best_epoch,
        "best_metric_name": BEST_METRIC,
        "best_metric_value": best_metric_value,
        "final_metrics": final_metrics,
        "final_per_label_f1": final_per_label_f1,
    }

    final_summary_file = (
        METRICS_DIR
        / "final_summary.json"
    )

    with final_summary_file.open(
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            final_summary,
            f,
            indent=2,
        )

    # ========================================================
    # Final summary
    # ========================================================

    print()
    print("=" * 70)
    print(
        "TRAINING + EVALUATION COMPLETED"
    )
    print("=" * 70)

    print(
        f"Training batches : "
        f"{global_batch_count:,}"
    )

    print(
        f"Training loss    : "
        f"{last_average_loss:.6f}"
    )

    print()
    print(
        f"Best epoch       : "
        f"{best_epoch}"
    )

    print(
        f"Best {BEST_METRIC:<12}: "
        f"{best_metric_value:.6f}"
    )

    print()
    print(
        f"Final Test Loss       : "
        f"{final_metrics['test_loss']:.6f}"
    )

    print(
        f"Final Micro F1        : "
        f"{final_metrics['micro_f1']:.6f}"
    )

    print(
        f"Final Macro F1        : "
        f"{final_metrics['macro_f1']:.6f}"
    )

    print(
        f"Final Weighted F1     : "
        f"{final_metrics['weighted_f1']:.6f}"
    )

    print(
        f"Final Micro Precision : "
        f"{final_metrics['micro_precision']:.6f}"
    )

    print(
        f"Final Macro Precision : "
        f"{final_metrics['macro_precision']:.6f}"
    )

    print(
        f"Final Micro Recall    : "
        f"{final_metrics['micro_recall']:.6f}"
    )

    print(
        f"Final Macro Recall    : "
        f"{final_metrics['macro_recall']:.6f}"
    )

    print(
        f"Final Hamming Loss    : "
        f"{final_metrics['hamming_loss']:.6f}"
    )

    print()
    print(
        f"Best model: "
        f"{best_checkpoint_dir}"
    )

    print(
        f"Epoch metrics: "
        f"{METRICS_DIR / 'epoch_metrics.csv'}"
    )

    print(
        f"Final per-label F1: "
        f"{final_per_label_file}"
    )

    print(
        f"Charts: "
        f"{PLOTS_DIR}"
    )

    print("=" * 70)


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":
    main()
