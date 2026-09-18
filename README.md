# Multilabel Document Classification

Multilabel document classification using **word graphs**, **centrality-based node selection**, and pretrained **BERT, FastText, and XLNet** models.

Datasets:

* AAPD
* RCV1

## Project Pipeline

```text
Raw Dataset
    ↓
Preprocessing
    ↓
Word Graph
    ↓
Closeness Centrality
    ↓
Top-K Nodes
    ↓
Sequence Extraction
    ↓
BERT / FastText / XLNet
    ↓
Multilabel Classification
```

## Installation

```powershell
python -m pip install --upgrade pip
pip install numpy pandas scikit-learn networkx nltk tqdm torch transformers gensim
```

## Dataset Preparation

### AAPD

```powershell
python src\prepare_aapd.py
```

Output:

```text
output/aapd/train.csv
output/aapd/test.csv
```

### RCV1

```powershell
python src\prepare_rcv1.py
```

The graph processor expects:

```text
output/rcv1-2/train.csv
output/rcv1-2/test.csv
```

If necessary:

```powershell
New-Item -ItemType Directory -Force output\rcv1-2
Copy-Item output\train.csv output\rcv1-2\train.csv -Force
Copy-Item output\test.csv output\rcv1-2\test.csv -Force
```

## Graph Processing

Default configuration:

* Window size: `3`
* Centrality: `closeness`
* Top-K nodes: `100`
* Maximum nodes: `25`
* Mode: `no_branch`

### AAPD

```powershell
python src\process_aapd_graphs.py --split train --workers 16 --window-size 3 --top-k 100 --max-nodes 25 --mode no_branch --chunksize 20

python src\process_aapd_graphs.py --split test --workers 16 --window-size 3 --top-k 100 --max-nodes 25 --mode no_branch --chunksize 20
```

### RCV1

```powershell
python src\process_rcv1_graphs.py --split train --workers 16 --window-size 3 --top-k 100 --max-nodes 25 --mode no_branch --chunksize 20

python src\process_rcv1_graphs.py --split test --workers 16 --window-size 3 --top-k 100 --max-nodes 25 --mode no_branch --chunksize 20
```

## Classification

### BERT

```powershell
python src\aapd_bert_linear.py
python src\rcv1_bert_linear.py
```

### FastText

```powershell
python src\aapd_fasttext_linear.py
python src\rcv1_fasttext_linear.py
```

### XLNet

```powershell
python src\appd_xlnet_linear.py
python src\rcv1_xlnet_linear.py
```

## Evaluation

The models are evaluated using:

* Precision
* Recall
* F1-score
* Hamming Loss

## Project Structure

```text
module/       Graph and sequence processing
raw/          Original datasets
pretrained/   Pretrained FastText model
output/       Processed data and trained models
src/          Data preparation, graph processing and classifiers
```

## Reproducibility

Main experimental configuration:

```text
Window Size = 3
Centrality = Closeness
Top-K = 100
Max Nodes = 25
Mode = no_branch
```
