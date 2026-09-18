from pathlib import Path
import csv
import string

import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize


# ============================================================
# Project paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

RAW_DIR = PROJECT_ROOT / "raw" / "aapd"
OUTPUT_DIR = PROJECT_ROOT / "output" / "aapd"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# NLTK resources
# ============================================================

def download_nltk_resources():
    resources = [
        ("tokenizers/punkt", "punkt"),
        ("tokenizers/punkt_tab", "punkt_tab"),
        ("corpora/stopwords", "stopwords"),
    ]

    for resource_path, resource_name in resources:
        try:
            nltk.data.find(resource_path)
        except LookupError:
            print(f"Downloading NLTK resource: {resource_name}")
            nltk.download(resource_name)


download_nltk_resources()


# ============================================================
# NLP tools
# ============================================================

STOP_WORDS = set(stopwords.words("english"))
STEMMER = PorterStemmer()
PUNCTUATION = set(string.punctuation)


# ============================================================
# Read raw files
# ============================================================

def read_lines(filename):
    path = RAW_DIR / filename

    if not path.exists():
        raise FileNotFoundError(
            f"File not found: {path}"
        )

    with open(path, "r", encoding="utf-8") as f:
        return [line.rstrip("\n") for line in f]


# ============================================================
# Label normalization
# ============================================================

def normalize_labels(label):
    """
    Convert AAPD label line into:

        label1|label2|label3

    Handles:
        spaces
        commas
        semicolons
        pipes
    """

    label = label.strip()

    if not label:
        return ""

    for separator in [",", ";"]:
        label = label.replace(separator, " ")

    labels = label.split()

    return "|".join(labels)


# ============================================================
# Get unique labels
# ============================================================

def get_unique_labels(labels):
    """
    Extract all unique labels from a list of raw/normalized
    label strings.
    """

    unique_labels = set()

    for label_line in labels:

        normalized = normalize_labels(label_line)

        if not normalized:
            continue

        for label in normalized.split("|"):
            if label:
                unique_labels.add(label)

    return unique_labels


# ============================================================
# Filter labels
# ============================================================

def filter_labels(labels, allowed_labels):
    """
    Keep only labels that exist in the training set.
    """

    normalized = normalize_labels(labels)

    if not normalized:
        return ""

    filtered = [
        label
        for label in normalized.split("|")
        if label in allowed_labels
    ]

    return "|".join(filtered)


# ============================================================
# Text preprocessing
# ============================================================

def preprocess_text(text):
    """
    Preprocessing:

    1. Lowercase
    2. NLTK tokenization
    3. Remove punctuation
    4. Remove stopwords
    5. Porter stemming
    """

    text = text.lower()

    tokens = word_tokenize(text)

    processed_tokens = []

    for token in tokens:

        # Remove punctuation
        if token in PUNCTUATION:
            continue

        if all(char in string.punctuation for char in token):
            continue

        # Remove stopwords
        if token in STOP_WORDS:
            continue

        # Porter stemming
        token = STEMMER.stem(token)

        if token:
            processed_tokens.append(token)

    return " ".join(processed_tokens)


# ============================================================
# Dataset preprocessing
# ============================================================

def preprocess_dataset(texts):
    processed_texts = []

    total = len(texts)

    for index, text in enumerate(texts, start=1):

        processed_texts.append(
            preprocess_text(text)
        )

        if index % 1000 == 0 or index == total:
            print(
                f"Processed {index:,}/{total:,} documents"
            )

    return processed_texts


# ============================================================
# Save unified CSV
# ============================================================

def save_csv(
    texts,
    labels,
    filename,
    split_prefix,
    allowed_labels=None
):
    """
    Save dataset using:

        doc_id,text,labels

    If allowed_labels is provided, only those labels
    are written.
    """

    assert len(texts) == len(labels), (
        f"Length mismatch: "
        f"{len(texts)} texts vs {len(labels)} labels"
    )

    output_path = OUTPUT_DIR / filename

    with open(
        output_path,
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

        for index, (text, label) in enumerate(
            zip(texts, labels),
            start=1
        ):

            # Stable document ID
            doc_id = f"{split_prefix}_{index:06d}"

            # ------------------------------------------------
            # Normalize labels
            # ------------------------------------------------

            normalized_label = normalize_labels(label)

            # ------------------------------------------------
            # For TEST:
            # keep only labels existing in TRAIN
            # ------------------------------------------------

            if allowed_labels is not None:

                normalized_label = filter_labels(
                    normalized_label,
                    allowed_labels
                )

            writer.writerow([
                doc_id,
                text,
                normalized_label
            ])

    print(f"Saved: {output_path}")


# ============================================================
# Read AAPD
# ============================================================

print("\n" + "=" * 60)
print("Reading AAPD dataset")
print("=" * 60)

text_train = read_lines("text_train")
label_train = read_lines("label_train")

text_validation = read_lines("text_validation")
label_validation = read_lines("label_validation")

text_test = read_lines("text_test")
label_test = read_lines("label_test")


# ============================================================
# Validate raw data
# ============================================================

assert len(text_train) == len(label_train), (
    "Train text/label length mismatch"
)

assert len(text_validation) == len(label_validation), (
    "Validation text/label length mismatch"
)

assert len(text_test) == len(label_test), (
    "Test text/label length mismatch"
)


print(f"Original train : {len(text_train):,}")
print(f"Validation     : {len(text_validation):,}")
print(f"Test           : {len(text_test):,}")


# ============================================================
# Preprocess Train
# ============================================================

print("\n" + "=" * 60)
print("Preprocessing train")
print("=" * 60)

text_train = preprocess_dataset(text_train)


# ============================================================
# Preprocess Validation
# ============================================================

print("\n" + "=" * 60)
print("Preprocessing validation")
print("=" * 60)

text_validation = preprocess_dataset(text_validation)


# ============================================================
# Preprocess Test
# ============================================================

print("\n" + "=" * 60)
print("Preprocessing test")
print("=" * 60)

text_test = preprocess_dataset(text_test)


# ============================================================
# Merge Train + Validation
# ============================================================

print("\n" + "=" * 60)
print("Merging train + validation")
print("=" * 60)

text_train_final = text_train + text_validation
label_train_final = label_train + label_validation


print(
    f"Final train size: {len(text_train_final):,}"
)


# ============================================================
# Find labels existing in TRAIN
# ============================================================

print("\n" + "=" * 60)
print("Finding labels present in TRAIN")
print("=" * 60)

train_labels = get_unique_labels(
    label_train_final
)

print(
    f"Unique labels in TRAIN: {len(train_labels)}"
)

print(
    "Train labels:"
)

print(
    sorted(train_labels)
)


# ============================================================
# Find TEST-only labels
# ============================================================

test_labels = get_unique_labels(label_test)

test_only_labels = test_labels - train_labels

print("\n" + "=" * 60)
print("Checking TEST-only labels")
print("=" * 60)

print(
    f"Unique labels in TEST       : {len(test_labels)}"
)

print(
    f"Labels also in TRAIN        : "
    f"{len(test_labels & train_labels)}"
)

print(
    f"TEST-only labels            : "
    f"{len(test_only_labels)}"
)

if test_only_labels:
    print("\nLabels that will be removed from TEST:")

    for label in sorted(test_only_labels):
        print(f"  - {label}")

else:
    print(
        "\nNo TEST-only labels found."
    )


# ============================================================
# Save TRAIN
# ============================================================

print("\n" + "=" * 60)
print("Saving TRAIN")
print("=" * 60)

save_csv(
    text_train_final,
    label_train_final,
    "train.csv",
    "AAPD_TRAIN"
)


# ============================================================
# Save TEST
# ============================================================

print("\n" + "=" * 60)
print("Saving TEST")
print("=" * 60)

save_csv(
    text_test,
    label_test,
    "test.csv",
    "AAPD_TEST",
    allowed_labels=train_labels
)


# ============================================================
# Summary
# ============================================================

print("\n" + "=" * 60)
print("Dataset summary")
print("=" * 60)

print(f"Original train       : {len(text_train):,}")
print(f"Validation           : {len(text_validation):,}")
print(f"Final train          : {len(text_train_final):,}")
print(f"Test                 : {len(text_test):,}")

print(
    f"\nUnique train labels  : {len(train_labels)}"
)

print(
    f"Unique test labels   : {len(test_labels)}"
)

print(
    f"Test-only labels     : {len(test_only_labels)}"
)

print("\nCSV schema:")
print("  doc_id,text,labels")

print("\nPreprocessing:")
print("  - lowercase")
print("  - NLTK word_tokenize")
print("  - punctuation removal")
print("  - English stopword removal")
print("  - Porter stemming")

print("\nLabel filtering:")
print("  - Train = train + validation")
print("  - Test labels not present in train are removed")

print("\nOutput directory:")
print(OUTPUT_DIR)