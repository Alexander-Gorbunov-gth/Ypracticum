from abc import ABC, abstractmethod
from transformers import BertTokenizerFast

from src.domain.constans import DataConstants


class iTokensPreparation(ABC):
    @abstractmethod
    def get_result(self, token: int) -> str:
        """Получить текст по токену"""

    @abstractmethod
    def get_tokens(self, text: str) -> list[int]:
        """Токенизирует текстовую последоватльеость"""


class TokensPreparation(iTokensPreparation):
    def __init__(self):
        self.tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")

    def get_result(self, tokens: list[int]) -> list[str]:
        """Получить текст по токену"""
        clean_tokens = [
            token
            for token in tokens
            if token not in [DataConstants.CLS_token, DataConstants.SEP_token]
        ]
        return self.tokenizer.decode(clean_tokens)

    def get_tokens(self, text: str) -> list[int]:
        """Токенизирует текстовую последоватльеость"""
        return self.tokenizer.encode(text)
