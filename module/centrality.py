from dataclasses import dataclass
from typing import Dict, List
import time

import networkx as nx


# ============================================================
# Supported strategies
# ============================================================

# CENTRALITY_STRATEGIES = [
#     "closeness",
#     "degree",
#     "betweenness",
#     "pagerank",
#     "clustering",
#     "closeness*degree*clustering",
# ]

CENTRALITY_STRATEGIES = [
    "closeness",
]


# ============================================================
# Result
# ============================================================

@dataclass
class StrategyResult:
    strategy: str
    scores: Dict[str, float]
    ranked_nodes: List[str]
    top_k_nodes: List[str]


# ============================================================
# Centrality Calculator
# ============================================================

class CentralityCalculator:

    def __init__(
        self,
        strategies: List[str] | None = None,
    ):

        self.strategies = (
            strategies
            if strategies is not None
            else CENTRALITY_STRATEGIES.copy()
        )

        unknown = (
            set(self.strategies)
            - set(CENTRALITY_STRATEGIES)
        )

        if unknown:
            raise ValueError(
                f"Unknown centrality strategies: "
                f"{sorted(unknown)}"
            )

    # ========================================================
    # Min-Max normalization
    # ========================================================

    @staticmethod
    def _min_max_normalize(scores):

        if not scores:
            return {}

        values = list(scores.values())

        min_value = min(values)
        max_value = max(values)

        if max_value == min_value:

            return {
                node: 0.0
                for node in scores
            }

        denominator = (
            max_value - min_value
        )

        return {
            node: (
                (score - min_value)
                / denominator
            )
            for node, score in scores.items()
        }

    # ========================================================
    # Rank nodes
    # ========================================================

    @staticmethod
    def _rank_nodes(scores):

        return sorted(
            scores,
            key=lambda node: (
                -scores[node],
                node,
            ),
        )

    # ========================================================
    # Calculate all centralities
    # ========================================================

    def calculate(self, graph):

        if not isinstance(graph, nx.Graph):
            raise TypeError(
                "graph must be a networkx.Graph"
            )

        results = {}

        # ----------------------------------------------------
        # IMPORTANT
        #
        # Calculate primitive centralities only once.
        #
        # The combined strategy will reuse these results.
        # ----------------------------------------------------

        if "closeness" in self.strategies:

            results["closeness"] = (
                nx.closeness_centrality(graph)
            )

        if "degree" in self.strategies:

            results["degree"] = (
                nx.degree_centrality(graph)
            )

        if "betweenness" in self.strategies:

            results["betweenness"] = (
                nx.betweenness_centrality(
                    graph,
                    normalized=True,
                )
            )

        if "pagerank" in self.strategies:

            results["pagerank"] = (
                nx.pagerank(
                    graph,
                    weight="weight",
                )
            )

        if "clustering" in self.strategies:

            results["clustering"] = (
                nx.clustering(
                    graph,
                    weight="weight",
                )
            )

        # ----------------------------------------------------
        # Combined strategy
        #
        # IMPORTANT:
        #
        # Reuse already calculated values.
        #
        # Do NOT call NetworkX again here.
        # ----------------------------------------------------

        if (
            "closeness*degree*clustering"
            in self.strategies
        ):

            # ------------------------------------------------
            # These values may not exist if the user requested
            # only the combined strategy.
            #
            # Calculate them only when necessary.
            # ------------------------------------------------

            if "closeness" not in results:

                results["closeness"] = (
                    nx.closeness_centrality(graph)
                )

            if "degree" not in results:

                results["degree"] = (
                    nx.degree_centrality(graph)
                )

            if "clustering" not in results:

                results["clustering"] = (
                    nx.clustering(
                        graph,
                        weight="weight",
                    )
                )

            closeness = (
                self._min_max_normalize(
                    results["closeness"]
                )
            )

            degree = (
                self._min_max_normalize(
                    results["degree"]
                )
            )

            clustering = (
                self._min_max_normalize(
                    results["clustering"]
                )
            )

            results[
                "closeness*degree*clustering"
            ] = {
                node:
                    closeness[node]
                    * degree[node]
                    * clustering[node]
                for node in graph.nodes
            }

        return results

    # ========================================================
    # Profiled calculation
    # ========================================================

    def calculate_profiled(self, graph):

        if not isinstance(graph, nx.Graph):
            raise TypeError(
                "graph must be a networkx.Graph"
            )

        results = {}
        timings = {}

        # ----------------------------------------------------
        # Closeness
        # ----------------------------------------------------

        if "closeness" in self.strategies:

            start = time.perf_counter()

            results["closeness"] = (
                nx.closeness_centrality(graph)
            )

            timings["closeness"] = (
                time.perf_counter() - start
            )

        # ----------------------------------------------------
        # Degree
        # ----------------------------------------------------

        if "degree" in self.strategies:

            start = time.perf_counter()

            results["degree"] = (
                nx.degree_centrality(graph)
            )

            timings["degree"] = (
                time.perf_counter() - start
            )

        # ----------------------------------------------------
        # Betweenness
        # ----------------------------------------------------

        if "betweenness" in self.strategies:

            start = time.perf_counter()

            results["betweenness"] = (
                nx.betweenness_centrality(
                    graph,
                    normalized=True,
                )
            )

            timings["betweenness"] = (
                time.perf_counter() - start
            )

        # ----------------------------------------------------
        # PageRank
        # ----------------------------------------------------

        if "pagerank" in self.strategies:

            start = time.perf_counter()

            results["pagerank"] = (
                nx.pagerank(
                    graph,
                    weight="weight",
                )
            )

            timings["pagerank"] = (
                time.perf_counter() - start
            )

        # ----------------------------------------------------
        # Clustering
        # ----------------------------------------------------

        if "clustering" in self.strategies:

            start = time.perf_counter()

            results["clustering"] = (
                nx.clustering(
                    graph,
                    weight="weight",
                )
            )

            timings["clustering"] = (
                time.perf_counter() - start
            )

        # ----------------------------------------------------
        # Combined strategy
        # ----------------------------------------------------
        #
        # IMPORTANT:
        #
        # We reuse:
        #
        #     results["closeness"]
        #     results["degree"]
        #     results["clustering"]
        #
        # Therefore the expensive NetworkX calculations are
        # NOT repeated.
        #
        # ----------------------------------------------------

        if (
            "closeness*degree*clustering"
            in self.strategies
        ):

            # ------------------------------------------------
            # In normal project configuration all three
            # primitive strategies are already calculated.
            #
            # The fallback below keeps the method robust if
            # someone initializes the calculator with only
            # the combined strategy.
            # ------------------------------------------------

            if "closeness" not in results:

                start = time.perf_counter()

                results["closeness"] = (
                    nx.closeness_centrality(graph)
                )

                timings["closeness"] = (
                    time.perf_counter() - start
                )

            if "degree" not in results:

                start = time.perf_counter()

                results["degree"] = (
                    nx.degree_centrality(graph)
                )

                timings["degree"] = (
                    time.perf_counter() - start
                )

            if "clustering" not in results:

                start = time.perf_counter()

                results["clustering"] = (
                    nx.clustering(
                        graph,
                        weight="weight",
                    )
                )

                timings["clustering"] = (
                    time.perf_counter() - start
                )

            start = time.perf_counter()

            closeness = (
                self._min_max_normalize(
                    results["closeness"]
                )
            )

            degree = (
                self._min_max_normalize(
                    results["degree"]
                )
            )

            clustering = (
                self._min_max_normalize(
                    results["clustering"]
                )
            )

            results[
                "closeness*degree*clustering"
            ] = {
                node:
                    closeness[node]
                    * degree[node]
                    * clustering[node]
                for node in graph.nodes
            }

            timings[
                "closeness*degree*clustering"
            ] = (
                time.perf_counter() - start
            )

        return results, timings

    # ========================================================
    # Rank
    # ========================================================

    def rank(
        self,
        graph,
        top_k=None,
    ):

        if (
            top_k is not None
            and top_k <= 0
        ):
            raise ValueError(
                "top_k must be positive"
            )

        centralities = self.calculate(
            graph
        )

        results = {}

        for strategy, scores in (
            centralities.items()
        ):

            ranked_nodes = (
                self._rank_nodes(scores)
            )

            if top_k is None:

                top_k_nodes = (
                    ranked_nodes.copy()
                )

            else:

                top_k_nodes = (
                    ranked_nodes[:top_k]
                )

            results[strategy] = StrategyResult(
                strategy=strategy,
                scores=scores,
                ranked_nodes=ranked_nodes,
                top_k_nodes=top_k_nodes,
            )

        return results