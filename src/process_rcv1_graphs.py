from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path


# ============================================================
# Project root
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================
# Project imports
# ============================================================

from module.centrality import (
    CentralityCalculator,
    StrategyResult,
)

from module.contracts import (
    ProcessedDocument,
    ProcessedToken,
)

from module.sequence_processor import (
    SequenceProcessor,
)

from module.word_graph import (
    WordGraphBuilder,
)

from tqdm import tqdm


INPUT_DIR = (
    PROJECT_ROOT
    / "output"
    / "rcv1-2"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "output"
    / "rcv1-2"
)


# ============================================================
# Worker globals
# ============================================================

WORD_GRAPH_BUILDER = None
CENTRALITY_CALCULATOR = None
SEQUENCE_PROCESSOR = None


# ============================================================
# Worker initialization
# ============================================================

def init_worker(
    window_size: int,
    max_nodes: int,
    mode: str,
):
    global WORD_GRAPH_BUILDER
    global CENTRALITY_CALCULATOR
    global SEQUENCE_PROCESSOR

    WORD_GRAPH_BUILDER = WordGraphBuilder(
        window_size=window_size
    )

    # Current experiment:
    # ONLY closeness.
    CENTRALITY_CALCULATOR = CentralityCalculator(
        strategies=["closeness"]
    )

    SEQUENCE_PROCESSOR = SequenceProcessor(
        max_nodes=max_nodes,
        mode=mode,
    )


# ============================================================
# Build document
# ============================================================

def build_document(
    row: dict[str, str],
) -> ProcessedDocument:

    text = row["text"]

    tokens = [
        ProcessedToken(
            text=word,
            position=position,
        )
        for position, word
        in enumerate(text.split())
    ]

    labels = (
        row["labels"].split("|")
        if row["labels"]
        else []
    )

    return ProcessedDocument(
        doc_id=str(row["doc_id"]),
        text=text,
        labels=labels,
        tokens=tokens,
    )


# ============================================================
# Create StrategyResult
# ============================================================

def create_strategy_results(
    centralities,
    top_k: int,
):

    results = {}

    for strategy, scores in (
        centralities.items()
    ):

        ranked_nodes = sorted(
            scores,
            key=lambda node: (
                -scores[node],
                node,
            ),
        )

        results[strategy] = StrategyResult(
            strategy=strategy,
            scores=scores,
            ranked_nodes=ranked_nodes,
            top_k_nodes=ranked_nodes[:top_k],
        )

    return results


# ============================================================
# Process one document
# ============================================================

def process_document(
    row: dict[str, str],
    top_k: int,
):

    document = build_document(
        row
    )

    word_graph = WORD_GRAPH_BUILDER.build(
        document
    )

    graph = word_graph.graph

    if graph.number_of_nodes() == 0:
        return []

    centralities = (
        CENTRALITY_CALCULATOR.calculate(
            graph
        )
    )

    strategy_results = (
        create_strategy_results(
            centralities,
            top_k,
        )
    )

    all_sequences = []

    for strategy_result in (
        strategy_results.values()
    ):

        sequences = (
            SEQUENCE_PROCESSOR.process(
                document=document,
                graph=graph,
                strategy_result=strategy_result,
            )
        )

        all_sequences.extend(
            sequences
        )

    # Convert dataclasses to tuples.
    return [
        (
            item.docid,
            item.sec,
            item.label,
        )
        for item in all_sequences
    ]


# ============================================================
# CSV reader
# ============================================================

def read_rows(
    input_file: Path,
):

    with input_file.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as f:

        reader = csv.DictReader(f)

        for row in reader:
            yield row


# ============================================================
# Count rows
# ============================================================

def count_rows(
    input_file: Path,
) -> int:

    with input_file.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as f:

        next(f, None)

        return sum(
            1
            for _ in f
        )


# ============================================================
# Main
# ============================================================

def process_split(
    split: str,
    workers: int,
    window_size: int,
    top_k: int,
    max_nodes: int,
    mode: str,
    chunksize: int,
):

    input_file = (
        INPUT_DIR
        / f"{split}.csv"
    )

    output_file = (
        OUTPUT_DIR
        / (
            f"{split}"
            f"_closeness"
            f"_{mode}"
            f"_top{top_k}"
            f"_nodes{max_nodes}"
            f".csv"
        )
    )

    if not input_file.exists():
        raise FileNotFoundError(
            input_file
        )

    total = count_rows(
        input_file
    )

    rows = read_rows(
        input_file
    )

    # IMPORTANT:
    # Do not use lambda here.
    # Windows multiprocessing requires
    # pickleable worker functions.
    worker = partial(
        process_document,
        top_k=top_k,
    )

    start = time.perf_counter()

    documents = 0
    sequences = 0

    output_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_file.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as f:

        writer = csv.writer(f)

        writer.writerow(
            [
                "docid",
                "sec",
                "label",
            ]
        )

        with ProcessPoolExecutor(
            max_workers=workers,
            initializer=init_worker,
            initargs=(
                window_size,
                max_nodes,
                mode,
            ),
        ) as executor:

            # map preserves document order.
            results = executor.map(
                worker,
                rows,
                chunksize=chunksize,
            )

            with tqdm(
                results,
                total=total,
                desc=f"Processing {split}",
                unit="doc",
                dynamic_ncols=True,
            ) as progress:

                for document_sequences in progress:

                    documents += 1

                    for (
                        docid,
                        sec,
                        label,
                    ) in document_sequences:

                        writer.writerow(
                            [
                                docid,
                                sec,
                                label,
                            ]
                        )

                        sequences += 1

    elapsed = (
        time.perf_counter()
        - start
    )

    throughput = (
        documents / elapsed
        if elapsed > 0
        else 0
    )

    print()
    print(
        f"Completed: "
        f"{documents:,} documents"
    )

    print(
        f"Sequences: "
        f"{sequences:,}"
    )

    print(
        f"Throughput: "
        f"{throughput:,.2f} docs/sec"
    )

    print(
        f"Output: "
        f"{output_file}"
    )


# ============================================================
# CLI
# ============================================================

def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--split",
        choices=[
            "train",
            "test",
        ],
        required=True,
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=max(
            1,
            (os.cpu_count() or 2) - 1,
        ),
    )

    parser.add_argument(
        "--window-size",
        type=int,
        default=3,
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=100,
    )

    parser.add_argument(
        "--max-nodes",
        type=int,
        default=25,
    )

    parser.add_argument(
        "--mode",
        choices=[
            "branch",
            "no_branch",
        ],
        default="no_branch",
    )

    parser.add_argument(
        "--chunksize",
        type=int,
        default=20,
    )

    return parser.parse_args()


# ============================================================
# Windows entry point
# ============================================================

def main():

    args = parse_args()

    process_split(
        split=args.split,
        workers=args.workers,
        window_size=args.window_size,
        top_k=args.top_k,
        max_nodes=args.max_nodes,
        mode=args.mode,
        chunksize=args.chunksize,
    )


if __name__ == "__main__":
    main()