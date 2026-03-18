from abc import ABC, abstractmethod
from typing import TextIO
import re
from pathlib import Path
from typing import Iterator, List
import pandas as pd


from src.services.tokens import iTokensPreparation
from src.config import cfg


class iDataPreparation(ABC):
    @abstractmethod
    def clean(self, text: str) -> str:
        """Удаляет лишние данные"""

    @abstractmethod
    def create_process_dataset(self, batch_size: int | None = None) -> str:
        """Запускает предварительную подготовку данных"""

    @abstractmethod
    def create_full_dataset(self, batch_size: int | None = None) -> str:
        """Запускает предварительную подготовку данных"""


class TextDataPreparation(iDataPreparation):
    LOCAL_DEVELOP_LINE_LIMIT = cfg.local_develop_line_limit

    def __init__(self, tokens_service: iTokensPreparation):
        self._tokens_service: iTokensPreparation = tokens_service()

    def clean(self, text):
        text = text.lower()
        text = re.sub(r"@\w+|https?://\S+", "", text)
        text = re.sub(r"[^a-z0-9\s]", "", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def _iter_blocks(self, source: TextIO) -> Iterator[str]:
        current: List[str] = []

        for raw in source:
            line = raw.strip()
            if not line:
                continue

            if line.startswith("@"):
                if current:
                    yield self.clean(" ".join(current))
                current = [line]
            else:
                if current:
                    current.append(line)
                else:
                    continue

        if current:
            yield " ".join(current)

    def _batched_blocks(
        self, source: TextIO, batch_size: int, min_text_length: int
    ) -> Iterator[list[str]]:
        batch = []
        for block in self._iter_blocks(source):
            if len(block.split(" ")) < min_text_length:
                continue
            # tokened_block = self._tokens_service.get_tokens(block)
            batch.append(block)
            if len(batch) >= batch_size:
                yield batch
                batch = []
        if batch:
            yield batch

    def create_process_dataset(
        self,
        input_path: Path,
        output_path: Path,
        batch_size: int = 1000,
        min_text_length: int = 4,
    ) -> str:
        if not self.input_path.is_file():
            raise ValueError("Отсутствуют исходные данные")

        if self.output_path.is_file():
            return "Документ уже существует, шаг пропущен"

        batch_count = 1

        with (
            input_path.open("r", encoding="utf-8") as input_file,
            output_path.open("w", encoding="utf-8", newline="") as output_file,
        ):
            for batch in self._batched_blocks(input_file, batch_size, min_text_length):
                df = pd.DataFrame(
                    {
                        "text": batch,
                    }
                )

                df.to_csv(
                    output_file,
                    header=False,
                    index=False,
                )
                if (
                    cfg.local_develop
                    and batch_count * batch_size >= cfg.local_develop_line_limit
                ):
                    break
                batch_count += 1
        return f"Создан новый файл csv с сырыми данными, количество строк - {batch_count * batch_size}"
