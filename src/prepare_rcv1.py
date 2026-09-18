from pathlib import Path
import csv
from collections import defaultdict


# ============================================================
# Paths
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

RCV1_DIR = BASE_DIR / "raw" / "rcv1"

QRELS_FILE = RCV1_DIR / "rcv1-v2.topics.qrels"

TRAIN_FILE = RCV1_DIR / "lyrl2004_tokens_train.dat"

TEST_FILES = [
    RCV1_DIR / "lyrl2004_tokens_test_pt0.dat",
    RCV1_DIR / "lyrl2004_tokens_test_pt1.dat",
    RCV1_DIR / "lyrl2004_tokens_test_pt2.dat",
    RCV1_DIR / "lyrl2004_tokens_test_pt3.dat",
]

OUTPUT_DIR = BASE_DIR / "output"


# ============================================================
# Load labels
# ============================================================

def load_labels(qrels_file):
    """
    Load RCV1 topic labels.

    File format:
        LABEL DOC_ID 1

    Example:
        E11 2286 1
        ECAT 2286 1
        M11 2286 1

    Returns:
        {
            "2286": ["E11", "ECAT", "M11"],
            ...
        }
    """

    labels = defaultdict(list)

    with open(qrels_file, "r", encoding="latin-1") as f:

        for line in f:
            parts = line.split()

            if len(parts) < 3:
                continue

            label = parts[0]
            doc_id = parts[1]
            relevance = parts[2]

            if relevance == "1":
                labels[doc_id].append(label)

    return dict(labels)


# ============================================================
# Read RCV1 token files
# ============================================================

def read_tokens(file_path):
    """
    Read an RCV1 token file.

    Expected structure:

        .I 2286
        .W
        word word word ...

    Returns:
        [(doc_id, text), ...]
    """

    documents = []

    current_id = None
    current_tokens = []
    reading_text = False

    with open(file_path, "r", encoding="latin-1") as f:

        for line in f:

            line = line.strip()

            if not line:
                continue

            # New document
            if line.startswith(".I"):

                if current_id is not None:
                    documents.append(
                        (
                            current_id,
                            " ".join(current_tokens)
                        )
                    )

                current_id = line.split()[1]
                current_tokens = []
                reading_text = False

            # Start of text section
            elif line == ".W":

                reading_text = True

            elif reading_text:

                current_tokens.extend(line.split())

    # Last document
    if current_id is not None:

        documents.append(
            (
                current_id,
                " ".join(current_tokens)
            )
        )

    return documents


# ============================================================
# Get labels that exist in TRAIN
# ============================================================

def get_train_labels(train_file, labels):
    """
    Find all labels that occur in the training documents.
    """

    train_labels = set()

    print("Finding labels present in TRAIN...")

    documents = read_tokens(train_file)

    for doc_id, _ in documents:

        for label in labels.get(doc_id, []):
            train_labels.add(label)

    return train_labels


# ============================================================
# Process files
# ============================================================

def process_files(
    input_files,
    labels,
    output_file,
    allowed_labels=None
):
    """
    Combine token files with labels and save as CSV.

    If allowed_labels is provided, only labels contained
    in allowed_labels will be written.
    """

    count = 0
    labeled_count = 0

    with open(
        output_file,
        "w",
        encoding="utf-8",
        newline=""
    ) as f:

        writer = csv.writer(f)

        writer.writerow([
            "doc_id",
            "text",
            "labels"
        ])

        for file_path in input_files:

            print(f"Reading: {file_path}")

            documents = read_tokens(file_path)

            for doc_id, text in documents:

                doc_labels = labels.get(doc_id, [])

                # ------------------------------------------------
                # Keep only labels that exist in TRAIN
                # ------------------------------------------------

                if allowed_labels is not None:

                    doc_labels = [
                        label
                        for label in doc_labels
                        if label in allowed_labels
                    ]

                writer.writerow([
                    doc_id,
                    text,
                    "|".join(doc_labels)
                ])

                count += 1

                if doc_labels:
                    labeled_count += 1

    print(f"Saved: {output_file}")
    print(f"Documents: {count}")
    print(f"Labeled: {labeled_count}")


# ============================================================
# Main
# ============================================================

def main():

    OUTPUT_DIR.mkdir(exist_ok=True)

    # ---------------------------------------------------------
    # Load all qrels labels
    # ---------------------------------------------------------

    print("Loading labels...")

    labels = load_labels(QRELS_FILE)

    print(
        f"Documents with labels: {len(labels)}"
    )

    # ---------------------------------------------------------
    # Find labels existing in TRAIN
    # ---------------------------------------------------------

    train_labels = get_train_labels(
        TRAIN_FILE,
        labels
    )

    print(
        f"Labels in TRAIN: {len(train_labels)}"
    )

    print(
        "Train labels:"
    )

    print(
        sorted(train_labels)
    )

    # ---------------------------------------------------------
    # TRAIN
    # ---------------------------------------------------------

    process_files(
        [TRAIN_FILE],
        labels,
        OUTPUT_DIR / "train.csv"
    )

    # ---------------------------------------------------------
    # TEST
    #
    # Only labels existing in TRAIN are kept.
    # ---------------------------------------------------------

    process_files(
        TEST_FILES,
        labels,
        OUTPUT_DIR / "test.csv",
        allowed_labels=train_labels
    )


if __name__ == "__main__":
    main()