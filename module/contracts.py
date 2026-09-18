
from dataclasses import dataclass
from typing import List


@dataclass(frozen=True)
class ProcessedToken:
    """
    A token after preprocessing.

    position:
        Original token position in the document.

    sentence_id:
        Identifier of the sentence containing this token.
    """

    text: str
    position: int

@dataclass
class ProcessedDocument:
    """
    Preprocessed document representation.
    """

    doc_id: str
    text: str
    labels: List[str]
    tokens: List[ProcessedToken]

    @property
    def token_count(self) -> int:
        return len(self.tokens)

    @property
    def token_texts(self) -> List[str]:
        return [token.text for token in self.tokens]

    @property
    def positions(self) -> List[int]:
        return [token.position for token in self.tokens]
