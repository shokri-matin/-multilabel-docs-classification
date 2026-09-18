from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import networkx as nx

from module.centrality import StrategyResult
from module.contracts import ProcessedDocument


ExtractionMode = Literal["branch", "no_branch"]


@dataclass
class ExtractedSequence:
    docid: str
    sec: str
    label: str


class SequenceProcessor:

    def __init__(
        self,
        max_nodes: int,
        mode: ExtractionMode,
    ):
        if max_nodes <= 0:
            raise ValueError(
                "max_nodes must be positive"
            )

        if mode not in {
            "branch",
            "no_branch",
        }:
            raise ValueError(
                "mode must be 'branch' or 'no_branch'"
            )

        self.max_nodes = max_nodes
        self.mode = mode

    # ========================================================
    # Graph preparation
    # ========================================================

    @staticmethod
    def _prepare_adjacency(
        graph: nx.Graph,
    ) -> dict[str, list[str]]:

        return {
            node: list(graph.neighbors(node))
            for node in graph.nodes
        }

    # ========================================================
    # Best neighbor
    # ========================================================

    @staticmethod
    def _best_neighbor(
        adjacency: dict[str, list[str]],
        central_node: str,
        scores: dict[str, float],
    ) -> str | None:

        neighbors = adjacency.get(
            central_node,
            []
        )

        if not neighbors:
            return None

        return min(
            neighbors,
            key=lambda node: (
                -scores[node],
                node,
            ),
        )

    # ========================================================
    # No branch traversal
    # ========================================================

    def _extract_no_branch(
        self,
        adjacency: dict[str, list[str]],
        central_node: str,
        best_neighbor: str,
    ) -> list[str]:

        selected = [
            central_node,
            best_neighbor,
        ]

        if self.max_nodes <= 2:
            return selected[:self.max_nodes]

        visited = set(selected)

        stack = [
            best_neighbor
        ]

        while (
            stack
            and len(selected) < self.max_nodes
        ):

            node = stack.pop()

            for neighbor in reversed(
                adjacency.get(node, [])
            ):

                if neighbor in visited:
                    continue

                visited.add(neighbor)

                selected.append(
                    neighbor
                )

                if len(selected) >= self.max_nodes:
                    break

                stack.append(neighbor)

        return selected

    # ========================================================
    # Branch traversal
    # ========================================================

    def _extract_branch(
        self,
        adjacency: dict[str, list[str]],
        central_node: str,
        best_neighbor: str,
    ) -> list[list[str]]:

        base_nodes = [
            central_node,
            best_neighbor,
        ]

        if self.max_nodes <= 2:
            return [
                base_nodes[:self.max_nodes]
            ]

        children = {
            best_neighbor: []
        }

        visited = {
            best_neighbor
        }

        stack = [
            best_neighbor
        ]

        while stack:

            node = stack.pop()

            for neighbor in reversed(
                adjacency.get(node, [])
            ):

                if neighbor in visited:
                    continue

                visited.add(neighbor)

                children.setdefault(
                    node,
                    []
                ).append(neighbor)

                children.setdefault(
                    neighbor,
                    []
                )

                stack.append(neighbor)

        root_children = [
            child
            for child in children.get(
                best_neighbor,
                []
            )
            if child != central_node
        ]

        if not root_children:
            return [base_nodes]

        branches = []

        for child in root_children:

            selected = base_nodes.copy()

            stack = [child]

            while (
                stack
                and len(selected)
                < self.max_nodes
            ):

                node = stack.pop()

                if node not in selected:
                    selected.append(node)

                for next_node in reversed(
                    children.get(node, [])
                ):
                    stack.append(next_node)

            branches.append(selected)

        return branches

    # ========================================================
    # Position mapping
    # ========================================================

    @staticmethod
    def _build_positions(
        document: ProcessedDocument,
    ) -> dict[str, list[int]]:

        positions: dict[str, list[int]] = {}

        for token in document.tokens:

            positions.setdefault(
                token.text,
                []
            ).append(
                token.position
            )

        return positions

    # ========================================================
    # Convert selected nodes → original sequence
    # ========================================================

    @staticmethod
    def _nodes_to_sequence(
        nodes,
        positions,
        tokens,
        max_nodes,
    ):
        selected_nodes = set(nodes)

        selected_positions = []

        for node in selected_nodes:
            for position in positions.get(node, []):
                selected_positions.append(position)

        selected_positions.sort()

        # Keep at most max_nodes tokens
        selected_positions = selected_positions[:max_nodes]

        sequence = [
            tokens[position].text
            for position in selected_positions
        ]

        # Padding
        while len(sequence) < max_nodes:
            sequence.append("<PAD>")

        return " ".join(sequence)

    # ========================================================
    # Process one strategy
    # ========================================================

    def process(
        self,
        document: ProcessedDocument,
        graph: nx.Graph,
        strategy_result: StrategyResult,
    ) -> list[ExtractedSequence]:

        adjacency = self._prepare_adjacency(
            graph
        )

        positions = self._build_positions(
            document
        )

        scores = strategy_result.scores

        results = []

        label = "|".join(
            document.labels
        )

        for central_node in (
            strategy_result.top_k_nodes
        ):

            best_neighbor = self._best_neighbor(
                adjacency,
                central_node,
                scores,
            )

            # ------------------------------------------------
            # Isolated node
            # ------------------------------------------------

            if best_neighbor is None:

                nodes = [
                    central_node
                ]

                sec = self._nodes_to_sequence(
                    nodes,
                    positions,
                    document.tokens,
                    self.max_nodes
                )

                if sec:

                    results.append(
                        ExtractedSequence(
                            docid=document.doc_id,
                            sec=sec,
                            label=label,
                        )
                    )

                continue

            # ------------------------------------------------
            # Traversal
            # ------------------------------------------------

            if self.mode == "no_branch":

                node_groups = [
                    self._extract_no_branch(
                        adjacency,
                        central_node,
                        best_neighbor,
                    )
                ]

            else:

                node_groups = self._extract_branch(
                    adjacency,
                    central_node,
                    best_neighbor,
                )

            # ------------------------------------------------
            # Convert every traversal to sequence
            # ------------------------------------------------

            for nodes in node_groups:

                sec = self._nodes_to_sequence(
                    nodes,
                    positions,
                    document.tokens,
                    self.max_nodes
                )

                if not sec:
                    continue

                results.append(
                    ExtractedSequence(
                        docid=document.doc_id,
                        sec=sec,
                        label=label,
                    )
                )

        return results