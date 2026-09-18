from collections import defaultdict
from dataclasses import dataclass

import networkx as nx

from module.contracts import ProcessedDocument

@dataclass
class WordGraph:
    doc_id: str
    graph: nx.Graph


class WordGraphBuilder:
    def __init__(self, window_size: int = 3):
        if window_size < 2:
            raise ValueError("window_size must be at least 2")

        self.window_size = window_size

    def build(self, document: ProcessedDocument) -> WordGraph:
        graph = nx.Graph()

        tokens = document.tokens

        # -----------------------------------------------------
        # Add unique words as nodes
        # -----------------------------------------------------

        positions_by_word = defaultdict(list)

        for token in tokens:
            positions_by_word[token.text].append(
                token.position
            )

        for word, positions in positions_by_word.items():
            graph.add_node(
                word,
                positions=sorted(positions),
            )

        # -----------------------------------------------------
        # Add weighted co-occurrence edges
        # -----------------------------------------------------

        for i in range(len(tokens)):
            source = tokens[i].text

            end = min(
                i + self.window_size,
                len(tokens),
            )

            for j in range(i + 1, end):
                target = tokens[j].text

                if source == target:
                    continue

                if graph.has_edge(source, target):
                    graph[source][target]["weight"] += 1
                else:
                    graph.add_edge(
                        source,
                        target,
                        weight=1,
                    )

        return WordGraph(
            doc_id=document.doc_id,
            graph=graph,
        )