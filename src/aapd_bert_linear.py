
from pathlib import Path

import csv
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

# ------------------------------------------------------------
# Maximum TOTAL training batches
#
# Example:
#
# 500  -> stop after 500 batches in total
# 1000 -> stop after 1000 batches in total
# None -> train for all batches in all epochs
# ------------------------------------------------------------

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


# ============================================================
# Checkpoint directory
# ============================================================

CHECKPOINT_DIR = Path(
    "output/models/bert_aapd"
)

CHECKPOINT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


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
    """
    Count the number of data rows in a CSV file.
    """

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
    """
    Count the total number of batches in a CSV file.
    """

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

        # ----------------------------------------------------
        # Last incomplete batch
        # ----------------------------------------------------

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
    """
    Convert pipe-separated labels
    into multi-hot vectors.
    """

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

        # ----------------------------------------------------
        # Load pretrained BERT
        # ----------------------------------------------------

        self.bert = AutoModel.from_pretrained(
            model_name
        )

        # ----------------------------------------------------
        # FREEZE BERT
        # ----------------------------------------------------

        for param in self.bert.parameters():
            param.requires_grad = False

        # ----------------------------------------------------
        # BERT hidden size
        # ----------------------------------------------------

        hidden_size = self.bert.config.hidden_size

        # ----------------------------------------------------
        # Two-layer trainable classifier
        # ----------------------------------------------------

        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, num_labels),
        )

    # ========================================================
    # Forward
    # ========================================================

    def forward(
        self,
        input_ids,
        attention_mask,
    ):

        # ----------------------------------------------------
        # BERT is frozen
        # ----------------------------------------------------

        with torch.no_grad():

            outputs = self.bert(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )

        # ----------------------------------------------------
        # CLS embedding
        # ----------------------------------------------------

        cls_embedding = outputs.last_hidden_state[:, 0]

        # ----------------------------------------------------
        # Two-layer classifier
        # ----------------------------------------------------

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
):
    """
    Save:

    1. Frozen pretrained BERT
    2. Tokenizer
    3. Trainable classifier
    4. Optimizer state
    5. Training configuration
    6. AAPD label vocabulary
    """

    checkpoint_dir = (
        CHECKPOINT_DIR
        / f"epoch_{epoch}_batch_{total_batches}"
    )

    checkpoint_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print()
    print(
        "Saving checkpoint to:"
    )

    print(
        checkpoint_dir
    )

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
    # Save training state
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

        "optimizer_state_dict":
            optimizer.state_dict(),
    }

    torch.save(
        training_state,
        checkpoint_dir / "training_state.pt",
    )

    print(
        "Checkpoint saved successfully."
    )

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
):
    """
    Evaluate trained classifier on AAPD test set.
    """

    print()
    print("=" * 70)
    print(
        "AAPD TEST EVALUATION"
    )
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

        for texts, labels in progress:

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
            # Targets
            # ------------------------------------------------

            targets = encode_labels(
                labels,
                label_to_id,
                num_labels,
            )

            targets = targets.to(
                DEVICE
            )

            # ------------------------------------------------
            # Forward
            # ------------------------------------------------

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

            total_loss += loss.item()

            batch_count += 1

            # ------------------------------------------------
            # Predictions
            # ------------------------------------------------

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

    # ========================================================
    # Concatenate
    # ========================================================

    y_pred = torch.cat(
        all_predictions,
        dim=0,
    ).numpy()

    y_true = torch.cat(
        all_targets,
        dim=0,
    ).numpy()

    # ========================================================
    # Test loss
    # ========================================================

    average_test_loss = (
        total_loss
        / max(1, batch_count)
    )

    # ========================================================
    # F1
    # ========================================================

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

    # ========================================================
    # Precision
    # ========================================================

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

    # ========================================================
    # Recall
    # ========================================================

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

    # ========================================================
    # Hamming Loss
    # ========================================================

    hamming = hamming_loss(
        y_true,
        y_pred,
    )

    # ========================================================
    # Results
    # ========================================================

    metrics = {

        "test_loss":
            average_test_loss,

        "micro_f1":
            micro_f1,

        "macro_f1":
            macro_f1,

        "weighted_f1":
            weighted_f1,

        "micro_precision":
            micro_precision,

        "macro_precision":
            macro_precision,

        "micro_recall":
            micro_recall,

        "macro_recall":
            macro_recall,

        "hamming_loss":
            hamming,
    }

    # ========================================================
    # Print results
    # ========================================================

    print()
    print("-" * 70)
    print(
        "TEST RESULTS"
    )
    print("-" * 70)

    print(
        f"Test Loss       : "
        f"{average_test_loss:.6f}"
    )

    print(
        f"Micro F1        : "
        f"{micro_f1:.6f}"
    )

    print(
        f"Macro F1        : "
        f"{macro_f1:.6f}"
    )

    print(
        f"Weighted F1     : "
        f"{weighted_f1:.6f}"
    )

    print(
        f"Micro Precision : "
        f"{micro_precision:.6f}"
    )

    print(
        f"Macro Precision : "
        f"{macro_precision:.6f}"
    )

    print(
        f"Micro Recall    : "
        f"{micro_recall:.6f}"
    )

    print(
        f"Macro Recall    : "
        f"{macro_recall:.6f}"
    )

    print(
        f"Hamming Loss    : "
        f"{hamming:.6f}"
    )

    print(
        f"Threshold       : "
        f"{THRESHOLD}"
    )

    print(
        f"Test samples    : "
        f"{len(y_true):,}"
    )

    print(
        f"Number of labels: "
        f"{num_labels}"
    )

    print("-" * 70)

    return metrics


# ============================================================
# Save evaluation results
# ============================================================

def save_evaluation_results(
    checkpoint_dir,
    metrics,
):
    """
    Save evaluation metrics as a text file.
    """

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

        f.write(
            "=" * 60
            + "\n"
        )

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
    print(
        "Evaluation results saved to:"
    )

    print(
        evaluation_file
    )


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

    print(
        f"Device: {DEVICE}"
    )

    print(
        f"Train CSV: {TRAIN_CSV}"
    )

    print(
        f"Test CSV: {TEST_CSV}"
    )

    print(
        f"Checkpoint directory: "
        f"{CHECKPOINT_DIR}"
    )

    print()
    print(
        "BERT training: DISABLED"
    )

    print(
        "Classifier training: ENABLED"
    )

    print()
    print(
        f"Batch size: {BATCH_SIZE}"
    )

    print(
        f"Epochs: {EPOCHS}"
    )

    print(
        f"Maximum training batches: "
        f"{MAX_TRAIN_BATCHES}"
    )

    print(
        f"Learning rate: {LEARNING_RATE}"
    )

    print(
        f"Threshold: {THRESHOLD}"
    )

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
    print(
        "Calculating training batches..."
    )

    total_dataset_batches = count_batches(
        TRAIN_CSV,
        BATCH_SIZE,
    )

    # --------------------------------------------------------
    # Determine total training batches
    # --------------------------------------------------------

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

    print()
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
    print(
        "Discovering AAPD labels..."
    )

    LABELS = discover_labels(
        TRAIN_CSV
    )

    NUM_LABELS = len(
        LABELS
    )

    LABEL_TO_ID = {
        label: idx
        for idx, label in enumerate(LABELS)
    }

    print(
        f"Number of labels: "
        f"{NUM_LABELS}"
    )

    print()
    print("Labels:")
    print(LABELS)

    # ========================================================
    # Tokenizer
    # ========================================================

    print()
    print(
        "Loading tokenizer..."
    )

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME
    )

    # ========================================================
    # Model
    # ========================================================

    print(
        "Loading BERT model..."
    )

    model = BertMultiLabelClassifier(
        MODEL_NAME,
        NUM_LABELS,
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
    print(
        "Trainable parameter check:"
    )

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
    # Training counters
    # ========================================================

    global_batch_count = 0

    training_finished = False

    last_epoch = 0

    last_average_loss = 0.0

    checkpoint_dir = None

    # ========================================================
    # Training
    # ========================================================

    for epoch in range(EPOCHS):

        last_epoch = epoch + 1

        model.train()

        # ----------------------------------------------------
        # BERT explicitly stays in evaluation mode
        # ----------------------------------------------------

        model.bert.eval()

        total_loss = 0.0

        epoch_batch_count = 0

        # ----------------------------------------------------
        # Epoch information
        # ----------------------------------------------------

        epochs_remaining = (
            EPOCHS
            - (epoch + 1)
        )

        batches_remaining_global = (
            total_training_batches
            - global_batch_count
        )

        print()
        print("=" * 70)

        print(
            f"Epoch {epoch + 1}/{EPOCHS}"
        )

        print(
            f"Epochs remaining: "
            f"{epochs_remaining}"
        )

        print(
            f"Global batches completed: "
            f"{global_batch_count}"
        )

        print(
            f"Global batches remaining: "
            f"{batches_remaining_global}"
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

        for texts, labels in progress:

            # ------------------------------------------------
            # Check maximum number of batches
            # ------------------------------------------------

            if (
                MAX_TRAIN_BATCHES is not None
                and global_batch_count
                >= MAX_TRAIN_BATCHES
            ):

                training_finished = True

                progress.close()

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
                labels,
                LABEL_TO_ID,
                NUM_LABELS,
            )

            targets = targets.to(
                DEVICE
            )

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

            # ------------------------------------------------
            # Remaining batches
            # ------------------------------------------------

            remaining_global_batches = (
                total_training_batches
                - global_batch_count
            )

            # ------------------------------------------------
            # Remaining batches in current epoch
            # ------------------------------------------------

            remaining_epoch_batches = max(
                0,
                total_dataset_batches
                - epoch_batch_count,
            )

            # ------------------------------------------------
            # Epochs remaining
            # ------------------------------------------------

            current_epochs_remaining = (
                EPOCHS
                - (epoch + 1)
            )

            # ------------------------------------------------
            # Global progress percentage
            # ------------------------------------------------

            progress_percent = (
                global_batch_count
                / total_training_batches
                * 100
            )

            # ------------------------------------------------
            # Update tqdm
            # ------------------------------------------------

            progress.set_postfix(
                done=(
                    f"{global_batch_count:,}"
                ),
                remaining=(
                    f"{remaining_global_batches:,}"
                ),
                epoch_done=(
                    f"{epoch_batch_count:,}"
                ),
                epoch_remaining=(
                    f"{remaining_epoch_batches:,}"
                ),
                epochs_left=(
                    f"{current_epochs_remaining:,}"
                ),
                progress=(
                    f"{progress_percent:.1f}%"
                ),
                loss=(
                    f"{loss_value:.4f}"
                ),
                avg_loss=(
                    f"{average_running_loss:.4f}"
                ),
            )

            # ------------------------------------------------
            # Stop after maximum batch limit
            # ------------------------------------------------

            if (
                MAX_TRAIN_BATCHES is not None
                and global_batch_count
                >= MAX_TRAIN_BATCHES
            ):

                training_finished = True

                progress.close()

                break

        # ====================================================
        # Epoch statistics
        # ====================================================

        average_loss = (
            total_loss
            / max(
                1,
                epoch_batch_count,
            )
        )

        last_average_loss = average_loss

        # ----------------------------------------------------
        # Remaining global batches
        # ----------------------------------------------------

        remaining_global_batches = (
            total_training_batches
            - global_batch_count
        )

        # ----------------------------------------------------
        # Overall progress
        # ----------------------------------------------------

        progress_percent = (
            global_batch_count
            / total_training_batches
            * 100
        )

        print()
        print(
            "-" * 70
        )

        print(
            f"Epoch {epoch + 1} completed"
        )

        print(
            f"Epoch batches completed : "
            f"{epoch_batch_count:,}"
        )

        print(
            f"Epoch batches total     : "
            f"{total_dataset_batches:,}"
        )

        print(
            f"Epoch batches remaining : "
            f"{max(0, total_dataset_batches - epoch_batch_count):,}"
        )

        print(
            f"Epochs remaining        : "
            f"{EPOCHS - (epoch + 1):,}"
        )

        print(
            f"Total batches completed : "
            f"{global_batch_count:,}"
        )

        print(
            f"Total batches remaining : "
            f"{remaining_global_batches:,}"
        )

        print(
            f"Overall progress        : "
            f"{progress_percent:.2f}%"
        )

        print(
            f"Average Loss            : "
            f"{average_loss:.6f}"
        )

        print(
            "-" * 70
        )

        # ====================================================
        # Maximum batch limit reached
        # ====================================================

        if training_finished:

            print()
            print(
                "=" * 70
            )

            print(
                "MAXIMUM TRAINING BATCHES REACHED"
            )

            print(
                f"Maximum batches: "
                f"{MAX_TRAIN_BATCHES:,}"
            )

            print(
                f"Actual batches: "
                f"{global_batch_count:,}"
            )

            print(
                f"Remaining batches: "
                f"{remaining_global_batches:,}"
            )

            print(
                f"Overall progress: "
                f"{progress_percent:.2f}%"
            )

            print(
                "Training stopped."
            )

            print(
                "=" * 70
            )

            # ------------------------------------------------
            # Save checkpoint
            # ------------------------------------------------

            checkpoint_dir = save_checkpoint(
                model=model,
                tokenizer=tokenizer,
                optimizer=optimizer,
                epoch=epoch + 1,
                total_batches=global_batch_count,
                average_loss=average_loss,
                labels=LABELS,
            )

            break

        # ====================================================
        # Normal epoch completion
        # ====================================================

        checkpoint_dir = save_checkpoint(
            model=model,
            tokenizer=tokenizer,
            optimizer=optimizer,
            epoch=epoch + 1,
            total_batches=global_batch_count,
            average_loss=average_loss,
            labels=LABELS,
        )

    # ========================================================
    # Training finished
    # ========================================================

    print()
    print("=" * 70)
    print(
        "TRAINING COMPLETED"
    )
    print("=" * 70)

    final_remaining_batches = (
        total_training_batches
        - global_batch_count
    )

    final_progress = (
        global_batch_count
        / total_training_batches
        * 100
    )

    print(
        f"Final epoch: "
        f"{last_epoch}"
    )

    print(
        f"Total training batches: "
        f"{global_batch_count:,}"
    )

    print(
        f"Remaining batches: "
        f"{final_remaining_batches:,}"
    )

    print(
        f"Overall progress: "
        f"{final_progress:.2f}%"
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
    # Evaluation
    # ========================================================

    metrics = evaluate(
        model=model,
        tokenizer=tokenizer,
        test_csv=TEST_CSV,
        label_to_id=LABEL_TO_ID,
        num_labels=NUM_LABELS,
    )

    # ========================================================
    # Save evaluation results
    # ========================================================

    if checkpoint_dir is not None:

        save_evaluation_results(
            checkpoint_dir=checkpoint_dir,
            metrics=metrics,
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

    print()
    print(
        f"Training batches : "
        f"{global_batch_count:,}"
    )

    print(
        f"Remaining batches: "
        f"{final_remaining_batches:,}"
    )

    print(
        f"Overall progress : "
        f"{final_progress:.2f}%"
    )

    print(
        f"Training loss    : "
        f"{last_average_loss:.6f}"
    )

    print()
    print(
        f"Test loss        : "
        f"{metrics['test_loss']:.6f}"
    )

    print(
        f"Micro F1         : "
        f"{metrics['micro_f1']:.6f}"
    )

    print(
        f"Macro F1         : "
        f"{metrics['macro_f1']:.6f}"
    )

    print(
        f"Weighted F1      : "
        f"{metrics['weighted_f1']:.6f}"
    )

    print(
        f"Micro Precision  : "
        f"{metrics['micro_precision']:.6f}"
    )

    print(
        f"Macro Precision  : "
        f"{metrics['macro_precision']:.6f}"
    )

    print(
        f"Micro Recall     : "
        f"{metrics['micro_recall']:.6f}"
    )

    print(
        f"Macro Recall     : "
        f"{metrics['macro_recall']:.6f}"
    )

    print(
        f"Hamming Loss     : "
        f"{metrics['hamming_loss']:.6f}"
    )

    print()
    print(
        f"Checkpoint: "
        f"{checkpoint_dir}"
    )

    print("=" * 70)


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":

    main()

