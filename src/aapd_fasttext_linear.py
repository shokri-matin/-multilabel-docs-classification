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


def document_vector(text):
    """
    Create a document embedding using mean pooling
    over pretrained FastText word embeddings.
    """

    words = text.split()

    if not words:
        return np.zeros(
            fasttext_model.vector_size,
            dtype=np.float32
        )

    vectors = [
        fasttext_model.get_vector(word)
        for word in words
    ]

    return np.mean(
        vectors,
        axis=0
    ).astype(np.float32)

# def document_vector(text):
#     """
#     Convert a document into a 300-dimensional
#     pretrained FastText vector.

#     FastText itself is NOT trained.
#     """

#     if not text.strip():
#         return np.zeros(
#             fasttext_model.vector_size,
#             dtype=np.float32
#         )

#     vector = fasttext_model.get_vector(
#         text
#     )

#     return np.asarray(
#         vector,
#         dtype=np.float32
#     )


# ============================================================
# Prepare Dataset
# ============================================================

def prepare_dataset(data):

    X = []
    Y = []

    print("\nCreating FastText document vectors...")

    for row in tqdm(data):

        vector = document_vector(
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

        X.append(vector)
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


print("\nTrain X shape:", X_train.shape)
print("Train y shape:", y_train.shape)

print("Test X shape:", X_test.shape)
print("Test y shape:", y_test.shape)


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
# Classifier
# ============================================================

class FastTextClassifier(nn.Module):

    def __init__(
        self,
        embedding_dim,
        num_labels
    ):

        super().__init__()

        self.classifier = nn.Linear(
            embedding_dim,
            num_labels
        )

    def forward(self, x):

        return self.classifier(x)


# ============================================================
# Model
# ============================================================

EMBEDDING_DIM = fasttext_model.vector_size

model = FastTextClassifier(
    embedding_dim=EMBEDDING_DIM,
    num_labels=NUM_LABELS
).to(DEVICE)


print("\nModel:")
print(model)


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
        f"Micro F1:       {metrics['micro_f1']:.4f}"
    )

    print(
        f"Macro F1:       {metrics['macro_f1']:.4f}"
    )

    print(
        f"Weighted F1:    {metrics['weighted_f1']:.4f}"
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
        f"Hamming Loss:   {metrics['hamming_loss']:.6f}"
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
            "embedding_dim": EMBEDDING_DIM,
            "num_labels": NUM_LABELS,
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