from abc import ABC, abstractmethod
from transformers import BertTokenizerFast
from functools import lru_cache


class iTokensPreparation(ABC):
    @abstractmethod
    def get_result(self, token: int) -> str:
        """Получить текст по токену"""

    @abstractmethod
    def get_tokens(self, text: str) -> list[int]:
        """Токенизирует текстовую последоватльеость"""

    @abstractmethod
    def decode_tokens(self, tokens: list[int]) -> str:
        """Декодирует список токенов в текст"""

    @abstractmethod
    def get_vocab_size(self) -> int:
        """Получить размер словаря"""


class TokensPreparation(iTokensPreparation):
    def __init__(self):
        self.tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")

    def get_result(self, token: int) -> str:
        """Получить текст по токену"""
        return self.tokenizer.decode(token)

    def get_tokens(self, text: str) -> list[int]:
        """Токенизирует текстовую последоватльеость"""
        return self.tokenizer.encode(text, add_special_tokens=False)

    def decode_tokens(self, tokens: list[int]) -> str:
        """Декодирует список токенов в текст"""
        return self.tokenizer.decode(
            tokens, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )

    def get_vocab_size(self) -> int:
        return self.tokenizer.vocab_size


@lru_cache
def get_tokens_service() -> iTokensPreparation:
    return TokensPreparation()


if __name__ == "__main__":
    service = get_tokens_service()
    print(service.get_vocab_size())
