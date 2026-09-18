from dataclasses import dataclass
from typing import Dict, List, Literal

import networkx as nx

from module.centrality import StrategyResult
from module.word_graph import WordGraph


ExtractionMode = Literal["branch", "no_branch"]


@dataclass
class SubgraphResult:
    doc_id: str | None
    strategy: str
    central_node: str
    central_score: float
    best_neighbor: str
    nodes: List[str]
    graph: nx.Graph


@dataclass
class LightweightSubgraphResult:
    doc_id: str | None
    strategy: str
    central_node: str
    central_score: float
    best_neighbor: str
    nodes: List[str]
    edges: List[tuple[str, str, float]]


@dataclass
class StrategySubgraphResult:
    strategy: str
    mode: ExtractionMode
    subgraphs: List[SubgraphResult]


@dataclass
class LightweightStrategySubgraphResult:
    strategy: str
    mode: ExtractionMode
    subgraphs: List[LightweightSubgraphResult]


class GraphProcessor:

    def __init__(self, max_nodes: int, mode: ExtractionMode):

        if max_nodes <= 0:
            raise ValueError("max_nodes must be positive")

        if mode not in {"branch", "no_branch"}:
            raise ValueError("mode must be 'branch' or 'no_branch'")

        self.max_nodes = max_nodes
        self.mode = mode

    # ---------------------------------------------------------
    # Graph preparation
    # ---------------------------------------------------------

    @staticmethod
    def _prepare_graph(graph):

        adjacency = {
            node: list(graph.neighbors(node))
            for node in graph.nodes
        }

        weights = {
            (u, v): data.get("weight", 1.0)
            for u, v, data in graph.edges(data=True)
        }

        return adjacency, weights

    # ---------------------------------------------------------
    # Best neighbor
    # ---------------------------------------------------------

    @staticmethod
    def _select_best_neighbor_from_adjacency(
        adjacency,
        central_node,
        scores,
    ):

        neighbors = adjacency.get(central_node)

        if not neighbors:
            return None

        best_node = None
        best_score = None

        for node in neighbors:

            score = scores[node]

            if (
                best_node is None
                or score > best_score
                or (score == best_score and node < best_node)
            ):
                best_node = node
                best_score = score

        return best_node

    # ---------------------------------------------------------
    # DFS
    # ---------------------------------------------------------

    @staticmethod
    def _dfs_preorder(adjacency, source):

        visited = {source}
        result = []

        stack = [source]

        while stack:

            node = stack.pop()

            result.append(node)

            neighbors = adjacency.get(node, [])

            for neighbor in reversed(neighbors):

                if neighbor not in visited:

                    visited.add(neighbor)
                    stack.append(neighbor)

        return result

    # ---------------------------------------------------------
    # No branch extraction
    # ---------------------------------------------------------

    def _extract_no_branch(
        self,
        adjacency,
        central_node,
        best_neighbor,
    ):

        selected = [central_node, best_neighbor]

        if self.max_nodes <= 2:
            return selected[:self.max_nodes]

        visited = set(selected)

        stack = [best_neighbor]

        while stack and len(selected) < self.max_nodes:

            node = stack.pop()

            for neighbor in reversed(adjacency.get(node, [])):

                if neighbor in visited:
                    continue

                visited.add(neighbor)
                selected.append(neighbor)

                if len(selected) >= self.max_nodes:
                    break

                stack.append(neighbor)

        return selected

    # ---------------------------------------------------------
    # Branch extraction
    # ---------------------------------------------------------

    def _extract_branch(
        self,
        adjacency,
        central_node,
        best_neighbor,
    ):

        base_nodes = [central_node, best_neighbor]

        if self.max_nodes <= 2:
            return [base_nodes[:self.max_nodes]]

        # Build DFS tree once.
        parent = {
            best_neighbor: None
        }

        children = {
            best_neighbor: []
        }

        visited = {
            best_neighbor
        }

        stack = [best_neighbor]

        while stack:

            node = stack.pop()

            for neighbor in reversed(adjacency.get(node, [])):

                if neighbor in visited:
                    continue

                visited.add(neighbor)

                parent[neighbor] = node

                children.setdefault(node, []).append(neighbor)
                children.setdefault(neighbor, [])

                stack.append(neighbor)

        branches = []

        root_children = [
            child
            for child in children.get(best_neighbor, [])
            if child != central_node
        ]

        if not root_children:
            return [base_nodes]

        for child in root_children:

            selected = base_nodes.copy()

            stack = [child]

            while stack and len(selected) < self.max_nodes:

                node = stack.pop()

                if node not in selected:
                    selected.append(node)

                for next_node in reversed(children.get(node, [])):
                    stack.append(next_node)

            branches.append(selected)

        return branches

    # ---------------------------------------------------------
    # Extract edges without creating NetworkX subgraph
    # ---------------------------------------------------------

    @staticmethod
    def _extract_edges(adjacency, weights, nodes):

        node_set = set(nodes)

        edges = []

        for source in nodes:

            for target in adjacency.get(source, []):

                if target not in node_set:
                    continue

                # Prevent duplicate undirected edges.
                if source < target:

                    weight = weights.get(
                        (source, target),
                        weights.get((target, source), 1.0),
                    )

                    edges.append(
                        (source, target, weight)
                    )

        return edges

    # ---------------------------------------------------------
    # Lightweight processing
    # ---------------------------------------------------------

    def process_strategy_lightweight(
        self,
        graph,
        strategy_result,
        doc_id=None,
    ):

        adjacency, weights = self._prepare_graph(graph)

        scores = strategy_result.scores

        subgraphs = []

        for central_node in strategy_result.top_k_nodes:

            central_score = scores[central_node]

            best_neighbor = (
                self._select_best_neighbor_from_adjacency(
                    adjacency,
                    central_node,
                    scores,
                )
            )

            # Isolated node
            if best_neighbor is None:

                nodes = [central_node]

                edges = []

                subgraphs.append(
                    LightweightSubgraphResult(
                        doc_id=doc_id,
                        strategy=strategy_result.strategy,
                        central_node=central_node,
                        central_score=central_score,
                        best_neighbor=central_node,
                        nodes=nodes,
                        edges=edges,
                    )
                )

                continue

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

            for nodes in node_groups:

                edges = self._extract_edges(
                    adjacency,
                    weights,
                    nodes,
                )

                subgraphs.append(
                    LightweightSubgraphResult(
                        doc_id=doc_id,
                        strategy=strategy_result.strategy,
                        central_node=central_node,
                        central_score=central_score,
                        best_neighbor=best_neighbor,
                        nodes=nodes,
                        edges=edges,
                    )
                )

        return LightweightStrategySubgraphResult(
            strategy=strategy_result.strategy,
            mode=self.mode,
            subgraphs=subgraphs,
        )

    # ---------------------------------------------------------
    # Process all strategies - lightweight
    # ---------------------------------------------------------

    def process_all_lightweight(
        self,
        graph,
        strategy_results,
        doc_id=None,
    ):

        return {
            strategy: self.process_strategy_lightweight(
                graph=graph,
                strategy_result=result,
                doc_id=doc_id,
            )
            for strategy, result in strategy_results.items()
        }

    # ---------------------------------------------------------
    # Original API
    # ---------------------------------------------------------

    def process_strategy(
        self,
        graph,
        strategy_result,
        doc_id=None,
    ):

        scores = strategy_result.scores

        subgraphs = []

        for central_node in strategy_result.top_k_nodes:

            central_score = scores[central_node]

            best_neighbor = self._select_best_neighbor_from_adjacency(
                {
                    node: list(graph.neighbors(node))
                    for node in graph.nodes
                },
                central_node,
                scores,
            )

            if best_neighbor is None:

                subgraph_graph = graph.subgraph(
                    [central_node]
                ).copy()

                subgraphs.append(
                    SubgraphResult(
                        doc_id=doc_id,
                        strategy=strategy_result.strategy,
                        central_node=central_node,
                        central_score=central_score,
                        best_neighbor=central_node,
                        nodes=[central_node],
                        graph=subgraph_graph,
                    )
                )

                continue

            adjacency = {
                node: list(graph.neighbors(node))
                for node in graph.nodes
            }

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

            for nodes in node_groups:

                subgraph_graph = graph.subgraph(
                    nodes
                ).copy()

                subgraphs.append(
                    SubgraphResult(
                        doc_id=doc_id,
                        strategy=strategy_result.strategy,
                        central_node=central_node,
                        central_score=central_score,
                        best_neighbor=best_neighbor,
                        nodes=nodes,
                        graph=subgraph_graph,
                    )
                )

        return StrategySubgraphResult(
            strategy=strategy_result.strategy,
            mode=self.mode,
            subgraphs=subgraphs,
        )

    def process_all(
        self,
        graph,
        strategy_results,
        doc_id=None,
    ):

        return {
            strategy: self.process_strategy(
                graph=graph,
                strategy_result=result,
                doc_id=doc_id,
            )
            for strategy, result in strategy_results.items()
        }