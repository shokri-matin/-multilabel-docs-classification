from pathlib import Path
from collections import Counter


# =========================
# Path
# =========================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "raw" / "aapd"

LABEL_FILE = RAW_DIR / "label_train"


# =========================
# Read labels
# =========================

counter = Counter()

with open(LABEL_FILE, "r", encoding="utf-8") as f:
    for line in f:
        labels = line.strip().split()
        counter.update(labels)


# =========================
# Sort by frequency
# =========================

total_documents = sum(counter.values())

sorted_labels = counter.most_common()


# =========================
# Print table
# =========================

print(f"{'Rank':<6} {'Label':<15} {'Frequency':>10} {'Percentage':>12}")
print("-" * 48)

for rank, (label, frequency) in enumerate(sorted_labels, start=1):

    percentage = (frequency / total_documents) * 100

    print(
        f"{rank:<6} "
        f"{label:<15} "
        f"{frequency:>10,} "
        f"{percentage:>11.2f}%"
    )


print("-" * 48)
print(f"Number of labels: {len(counter)}")
print(f"Total label assignments: {total_documents:,}")