from pathlib import Path
from collections import Counter


# =========================
# Paths
# =========================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "raw" / "aapd"


# =========================
# Read labels
# =========================

def read_labels(filename):
    path = RAW_DIR / filename

    with open(path, "r", encoding="utf-8") as f:
        return [line.strip().split() for line in f if line.strip()]


# =========================
# Analyze split
# =========================

def analyze_split(name, filename):
    labels_per_doc = read_labels(filename)

    # Number of labels for each document
    num_labels = [len(labels) for labels in labels_per_doc]

    # Label frequency
    label_counter = Counter(
        label
        for labels in labels_per_doc
        for label in labels
    )

    print(f"\n{'=' * 50}")
    print(f"{name}")
    print(f"{'=' * 50}")

    print(f"Documents              : {len(labels_per_doc):,}")
    print(f"Unique labels          : {len(label_counter):,}")

    if num_labels:
        print(f"Average labels/doc     : {sum(num_labels) / len(num_labels):.2f}")
        print(f"Min labels/doc         : {min(num_labels)}")
        print(f"Max labels/doc         : {max(num_labels)}")

    print("\nLabels per document:")
    distribution = Counter(num_labels)

    for n_labels in sorted(distribution):
        count = distribution[n_labels]
        percentage = count / len(num_labels) * 100

        print(
            f"  {n_labels:2d} labels : "
            f"{count:6,} documents "
            f"({percentage:6.2f}%)"
        )

    print("\nLabel frequency:")
    for label, count in label_counter.most_common():
        percentage = count / len(labels_per_doc) * 100

        print(
            f"  {label:15s} : "
            f"{count:6,} documents "
            f"({percentage:6.2f}%)"
        )

    return label_counter


# =========================
# Main
# =========================

train_counter = analyze_split(
    "TRAIN",
    "label_train"
)

validation_counter = analyze_split(
    "VALIDATION",
    "label_validation"
)

test_counter = analyze_split(
    "TEST",
    "label_test"
)


# =========================
# Overall label distribution
# =========================

all_labels = train_counter + validation_counter + test_counter

print(f"\n{'=' * 50}")
print("ALL DATA")
print(f"{'=' * 50}")

print(f"Total unique labels : {len(all_labels)}")

print("\nOverall label frequency:")

for label, count in all_labels.most_common():
    print(f"  {label:15s} : {count:6,}")


# =========================
# Imbalance analysis
# =========================

print(f"\n{'=' * 50}")
print("IMBALANCE ANALYSIS")
print(f"{'=' * 50}")

frequencies = list(all_labels.values())

max_freq = max(frequencies)
min_freq = min(frequencies)

print(f"Most frequent label : {max_freq:,}")
print(f"Least frequent label: {min_freq:,}")

print(f"Imbalance Ratio     : {max_freq / min_freq:.2f}")

print("\nRare labels:")

for threshold in [10, 50, 100, 500]:
    count = sum(
        1 for freq in frequencies
        if freq <= threshold
    )

    print(
        f"Labels with <= {threshold:4d} documents: "
        f"{count}"
    )