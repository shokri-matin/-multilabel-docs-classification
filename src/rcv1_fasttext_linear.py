from pathlib import Path

import csv
import numpy as np
import torch
import torch.nn as nn

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
    "output/rcv1-2/train_closeness_no_branch_top100_nodes25.csv"
)

TEST_CSV = Path(
    "output/rcv1-2/test_closeness_no_branch_top100_nodes25.csv"
)


# ============================================================
# Pretrained FastText
# ============================================================

FASTTEXT_MODEL_PATH = Path(
    "pretrained/fasttext/crawl-300d-2M-subword.bin"
)


# ============================================================
# Training configuration
# ============================================================

BATCH_SIZE = 32

EPOCHS = 10

# Maximum TOTAL training batches.
#
# 5    -> only 5 batches in total
# 500  -> only 500 batches in total
# None -> all batches in all epochs

MAX_TRAIN_BATCHES = 5

LEARNING_RATE = 2e-4

DROPOUT = 0.1

THRESHOLD = 0.5


# ============================================================
# Checkpoint directory
# ============================================================

CHECKPOINT_DIR = Path(
    "output/models/fasttext_rcv1"
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
# Official RCV1-v2 Topic Labels
# ============================================================

LABELS = [
    "C11",
    "C12",
    "C13",
    "C14",
    "C15",
    "C151",
    "C1511",
    "C152",
    "C16",
    "C17",
    "C171",
    "C172",
    "C173",
    "C174",
    "C18",
    "C181",
    "C182",
    "C183",
    "C21",
    "C22",
    "C23",
    "C24",
    "C31",
    "C311",
    "C312",
    "C313",
    "C32",
    "C33",
    "C331",
    "C34",
    "C41",
    "C411",
    "C42",
    "CCAT",
    "E11",
    "E12",
    "E121",
    "E13",
    "E131",
    "E132",
    "E14",
    "E141",
    "E142",
    "E143",
    "E21",
    "E211",
    "E212",
    "E31",
    "E311",
    "E312",
    "E313",
    "E41",
    "E411",
    "E51",
    "E511",
    "E512",
    "E513",
    "E61",
    "E71",
    "ECAT",
    "G15",
    "G151",
    "G152",
    "G153",
    "G154",
    "G155",
    "G156",
    "G157",
    "G158",
    "G159",
    "GCAT",
    "GCRIM",
    "GDEF",
    "GDIP",
    "GDIS",
    "GENT",
    "GENV",
    "GFAS",
    "GHEA",
    "GJOB",
    "GMIL",
    "GOBIT",
    "GODD",
    "GPOL",
    "GPRO",
    "GREL",
    "GSCI",
    "GSPO",
    "GTOUR",
    "GVIO",
    "GVOTE",
    "GWEA",
    "GWELF",
    "M11",
    "M12",
    "M13",
    "M131",
    "M132",
    "M14",
    "M141",
    "M142",
    "M143",
    "MCAT",
]


NUM_LABELS = len(LABELS)

LABEL_TO_ID = {
    label: idx
    for idx, label in enumerate(LABELS)
}

assert NUM_LABELS == 103


# ============================================================
# Load FastText
# ============================================================

def load_fasttext_model(
    model_path,
):
    """
    Load pretrained FastText vectors.

    The FastText model is used only for generating
    fixed word embeddings.

    It is NOT fine-tuned.
    """

    print()
    print(
        "Loading pretrained FastText..."
    )

    print(
        f"Model path: {model_path}"
    )

    if not model_path.exists():

        raise FileNotFoundError(
            f"FastText model not found:\n"
            f"{model_path}"
        )

    fasttext_model = load_facebook_vectors(
        str(model_path)
    )

    print(
        "FastText loaded successfully."
    )

    print(
        f"Vector dimension: "
        f"{fasttext_model.vector_size}"
    )

    return fasttext_model


# ============================================================
# Count CSV rows
# ============================================================

def count_csv_rows(
    csv_file,
):
    """
    Count data rows in a CSV file.
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
    Calculate number of batches.

    The final incomplete batch is included.
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

    Expected schema:

        docid,sec,label

    'sec' contains graph/subgraph text.
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
                    f"Unknown RCV1 label: "
                    f"{label}"
                )

            label_id = label_to_id[label]

            target[
                row_idx,
                label_id
            ] = 1.0

    return target


# ============================================================
# FastText document embedding
# ============================================================

def fasttext_document_vector(
    fasttext_model,
    text,
):
    """
    Convert one document into one FastText vector.

    Pipeline:

        document
            |
            v
        words
            |
            v
        FastText word vectors
            |
            v
        Mean Pooling
            |
            v
        document embedding

    IMPORTANT:

    The entire document is NOT passed to FastText
    as one token.

    Each word is embedded separately.
    """

    words = text.split()

    embedding_dim = (
        fasttext_model.vector_size
    )

    # --------------------------------------------------------
    # Empty document
    # --------------------------------------------------------

    if not words:

        return np.zeros(
            embedding_dim,
            dtype=np.float32,
        )

    # --------------------------------------------------------
    # Word vectors
    # --------------------------------------------------------

    vectors = np.asarray(
        [
            fasttext_model.get_vector(
                word
            )
            for word in words
        ],
        dtype=np.float32,
    )

    # --------------------------------------------------------
    # Mean Pooling
    # --------------------------------------------------------

    document_vector = vectors.mean(
        axis=0
    )

    return document_vector.astype(
        np.float32
    )


# ============================================================
# Batch FastText document embeddings
# ============================================================

def texts_to_embeddings(
    fasttext_model,
    texts,
):
    """
    Convert a batch of documents into
    document-level FastText embeddings.

    Output:

        [batch_size, embedding_dim]
    """

    embeddings = np.stack(
        [
            fasttext_document_vector(
                fasttext_model,
                text,
            )
            for text in texts
        ]
    )

    return torch.from_numpy(
        embeddings
    )


# ============================================================
# FastText Multi-Label Classifier
# ============================================================

class FastTextMultiLabelClassifier(
    nn.Module
):
    """
    Frozen FastText document embeddings
    + trainable linear classifier.

    Architecture:

        text
          |
          v
        words
          |
          v
        FastText
          |
          v
        Mean Pooling
          |
          v
        document vector
          |
          v
        Dropout
          |
          v
        Linear
          |
          v
        103 logits
    """

    def __init__(
        self,
        embedding_dim,
        num_labels,
        dropout=0.1,
    ):

        super().__init__()

        self.embedding_dim = (
            embedding_dim
        )

        self.num_labels = (
            num_labels
        )

        # ----------------------------------------------------
        # Dropout
        # ----------------------------------------------------

        self.dropout = nn.Dropout(
            dropout
        )

        # ----------------------------------------------------
        # Trainable classifier
        # ----------------------------------------------------

        self.classifier = nn.Linear(
            embedding_dim,
            num_labels,
        )

    def forward(
        self,
        document_embeddings,
    ):

        embeddings = self.dropout(
            document_embeddings
        )

        logits = self.classifier(
            embeddings
        )

        return logits


# ============================================================
# Save checkpoint
# ============================================================

def save_checkpoint(
    model,
    optimizer,
    epoch,
    total_batches,
    average_loss,
    embedding_dim,
):
    """
    Save classifier and training state.

    The pretrained FastText model is NOT copied.
    Only its path is stored.
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
    # Classifier
    # --------------------------------------------------------

    torch.save(
        model.classifier.state_dict(),
        checkpoint_dir / "classifier.pt",
    )

    # --------------------------------------------------------
    # Training state
    # --------------------------------------------------------

    training_state = {

        "epoch":
            epoch,

        "total_batches":
            total_batches,

        "average_loss":
            average_loss,

        "learning_rate":
            LEARNING_RATE,

        "batch_size":
            BATCH_SIZE,

        "max_train_batches":
            MAX_TRAIN_BATCHES,

        "threshold":
            THRESHOLD,

        "num_labels":
            NUM_LABELS,

        "labels":
            LABELS,

        "embedding_dim":
            embedding_dim,

        "dropout":
            DROPOUT,

        "fasttext_model_path":
            str(FASTTEXT_MODEL_PATH),

        "fasttext_frozen":
            True,

        "pooling":
            "word_mean_pooling",

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
    fasttext_model,
    test_csv,
    label_to_id,
    num_labels,
):
    """
    Evaluate trained FastText classifier.
    """

    print()
    print("=" * 70)
    print(
        "RCV1 FASTTEXT TEST EVALUATION"
    )
    print("=" * 70)

    model.eval()

    all_predictions = []
    all_targets = []

    criterion = nn.BCEWithLogitsLoss()

    total_loss = 0.0
    batch_count = 0

    # --------------------------------------------------------
    # Count test batches
    # --------------------------------------------------------

    total_test_batches = count_batches(
        test_csv,
        BATCH_SIZE,
    )

    print(
        f"Total test batches: "
        f"{total_test_batches:,}"
    )

    batches = read_batches(
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

    # ========================================================
    # Evaluation loop
    # ========================================================

    with torch.no_grad():

        for texts, labels in progress:

            # ------------------------------------------------
            # FastText document embeddings
            # ------------------------------------------------

            document_embeddings = (
                texts_to_embeddings(
                    fasttext_model,
                    texts,
                )
                .to(DEVICE)
            )

            # ------------------------------------------------
            # Targets
            # ------------------------------------------------

            targets = encode_labels(
                labels,
                label_to_id,
                num_labels,
            ).to(DEVICE)

            # ------------------------------------------------
            # Forward
            # ------------------------------------------------

            logits = model(
                document_embeddings
            )

            # ------------------------------------------------
            # Loss
            # ------------------------------------------------

            loss = criterion(
                logits,
                targets,
            )

            total_loss += (
                loss.item()
            )

            batch_count += 1

            # ------------------------------------------------
            # Probabilities
            # ------------------------------------------------

            probabilities = torch.sigmoid(
                logits
            )

            # ------------------------------------------------
            # Predictions
            # ------------------------------------------------

            predictions = (
                probabilities >= THRESHOLD
            ).float()

            all_predictions.append(
                predictions.cpu()
            )

            all_targets.append(
                targets.cpu()
            )

            # ------------------------------------------------
            # Progress
            # ------------------------------------------------

            remaining_batches = (
                total_test_batches
                - batch_count
            )

            progress_percent = (
                batch_count
                / total_test_batches
                * 100
            )

            progress.set_postfix(
                done=(
                    f"{batch_count:,}"
                ),
                remaining=(
                    f"{remaining_batches:,}"
                ),
                progress=(
                    f"{progress_percent:.1f}%"
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
        / max(
            1,
            batch_count,
        )
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
    # Metrics
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
    Save evaluation metrics.
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
            "RCV1 FastText Evaluation Results\n"
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
            f"max_train_batches: "
            f"{MAX_TRAIN_BATCHES}\n"
        )

        f.write(
            f"num_labels: "
            f"{NUM_LABELS}\n"
        )

        f.write(
            f"fasttext_model: "
            f"{FASTTEXT_MODEL_PATH}\n"
        )

        f.write(
            "pooling: word_mean_pooling\n"
        )

        f.write(
            "fasttext_frozen: True\n"
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
        "RCV1 FASTTEXT Multi-Label Training"
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
        f"FastText model: "
        f"{FASTTEXT_MODEL_PATH}"
    )

    print(
        f"Checkpoint directory: "
        f"{CHECKPOINT_DIR}"
    )

    print()

    print(
        "FastText training: DISABLED"
    )

    print(
        "Classifier training: ENABLED"
    )

    print()

    print(
        f"Batch size: "
        f"{BATCH_SIZE}"
    )

    print(
        f"Epochs: "
        f"{EPOCHS}"
    )

    print(
        f"Maximum training batches: "
        f"{MAX_TRAIN_BATCHES}"
    )

    print(
        f"Learning rate: "
        f"{LEARNING_RATE}"
    )

    print(
        f"Dropout: "
        f"{DROPOUT}"
    )

    print(
        f"Threshold: "
        f"{THRESHOLD}"
    )

    print(
        "Pooling: Word-level Mean Pooling"
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

    if not FASTTEXT_MODEL_PATH.exists():

        raise FileNotFoundError(
            f"FastText model not found:\n"
            f"{FASTTEXT_MODEL_PATH}"
        )

    # ========================================================
    # Count training batches
    # ========================================================

    print()
    print(
        "Calculating training batches..."
    )

    total_dataset_batches = count_batches(
        TRAIN_CSV,
        BATCH_SIZE,
    )

    total_possible_training_batches = (
        total_dataset_batches
        * EPOCHS
    )

    if MAX_TRAIN_BATCHES is None:

        total_training_batches = (
            total_possible_training_batches
        )

    else:

        total_training_batches = min(
            MAX_TRAIN_BATCHES,
            total_possible_training_batches,
        )

    print()

    print(
        f"Training batches per epoch : "
        f"{total_dataset_batches:,}"
    )

    print(
        f"Total possible batches     : "
        f"{total_possible_training_batches:,}"
    )

    print(
        f"Total training batches     : "
        f"{total_training_batches:,}"
    )

    # ========================================================
    # Labels
    # ========================================================

    label_to_id = LABEL_TO_ID

    num_labels = len(
        label_to_id
    )

    print()

    print(
        f"Number of labels: "
        f"{num_labels}"
    )

    if num_labels != 103:

        raise ValueError(
            f"Expected 103 RCV1 labels, "
            f"found {num_labels}"
        )

    # ========================================================
    # Load FastText
    # ========================================================

    fasttext_model = load_fasttext_model(
        FASTTEXT_MODEL_PATH
    )

    embedding_dim = (
        fasttext_model.vector_size
    )

    # ========================================================
    # Model
    # ========================================================

    print()
    print(
        "Creating FastText classifier..."
    )

    model = FastTextMultiLabelClassifier(
        embedding_dim=embedding_dim,
        num_labels=num_labels,
        dropout=DROPOUT,
    )

    model.to(DEVICE)

    # ========================================================
    # Parameter check
    # ========================================================

    trainable_params = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    print()

    print(
        "Trainable parameter check:"
    )

    print(
        f"Classifier trainable parameters: "
        f"{trainable_params:,}"
    )

    expected_trainable_params = (
        embedding_dim * num_labels
        + num_labels
    )

    print(
        f"Expected classifier parameters: "
        f"{expected_trainable_params:,}"
    )

    if trainable_params == 0:

        raise RuntimeError(
            "Classifier has no trainable parameters!"
        )

    if trainable_params != expected_trainable_params:

        raise RuntimeError(
            "Unexpected number of trainable parameters!"
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

        total_loss = 0.0

        epoch_batch_count = 0

        epochs_remaining = (
            EPOCHS
            - (epoch + 1)
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
            f"{global_batch_count:,}"
        )

        print(
            f"Global batches remaining: "
            f"{max(0, total_training_batches - global_batch_count):,}"
        )

        print("=" * 70)

        batches = read_batches(
            TRAIN_CSV,
            BATCH_SIZE,
        )

        progress = tqdm(
            batches,
            total=total_dataset_batches,
            desc=f"Epoch {epoch + 1}/{EPOCHS}",
            unit="batch",
            dynamic_ncols=True,
        )

        # ====================================================
        # Batch loop
        # ====================================================

        for texts, labels in progress:

            # ------------------------------------------------
            # Maximum batch check
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
            # Create document embeddings
            # ------------------------------------------------

            document_embeddings = (
                texts_to_embeddings(
                    fasttext_model,
                    texts,
                )
                .to(DEVICE)
            )

            # ------------------------------------------------
            # Labels
            # ------------------------------------------------

            targets = encode_labels(
                labels,
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
                document_embeddings
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
            # Remaining global batches
            # ------------------------------------------------

            remaining_global_batches = (
                total_training_batches
                - global_batch_count
            )

            # ------------------------------------------------
            # Remaining epoch batches
            # ------------------------------------------------

            remaining_epoch_batches = max(
                0,
                total_dataset_batches
                - epoch_batch_count,
            )

            # ------------------------------------------------
            # Remaining epochs
            # ------------------------------------------------

            current_epochs_remaining = (
                EPOCHS
                - (epoch + 1)
            )

            # ------------------------------------------------
            # Overall progress
            # ------------------------------------------------

            progress_percent = (
                global_batch_count
                / total_training_batches
                * 100
            )

            # ------------------------------------------------
            # Progress bar
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
            # Stop after maximum batches
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

        remaining_global_batches = (
            total_training_batches
            - global_batch_count
        )

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
                optimizer=optimizer,
                epoch=epoch + 1,
                total_batches=global_batch_count,
                average_loss=average_loss,
                embedding_dim=embedding_dim,
            )

            break

        # ====================================================
        # Normal epoch completion
        # ====================================================

        checkpoint_dir = save_checkpoint(
            model=model,
            optimizer=optimizer,
            epoch=epoch + 1,
            total_batches=global_batch_count,
            average_loss=average_loss,
            embedding_dim=embedding_dim,
        )

    # ========================================================
    # Training completed
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
        "FastText remained pretrained/frozen."
    )

    print(
        "Only the classifier was trained."
    )

    # ========================================================
    # Evaluation
    # ========================================================

    metrics = evaluate(
        model=model,
        fasttext_model=fasttext_model,
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

