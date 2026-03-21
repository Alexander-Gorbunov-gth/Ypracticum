from abc import ABC, abstractmethod
from typing import TextIO
import re
from pathlib import Path
from typing import Iterator, List
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm


from src.services.tokens import iTokensPreparation, get_tokens_service
from src.config import cfg
from src.domain.constans import DataConstants


class iDataPreparation(ABC):
    @abstractmethod
    def clean(self, text: str) -> str:
        """Удаляет лишние данные"""

    @abstractmethod
    def create_process_dataset(self, batch_size: int | None = None) -> str:
        """Запускает предварительную подготовку данных"""

    @abstractmethod
    def create_next_token_dataset(
        self, input_path: Path, output_path: Path, batch_size: int = 1000
    ) -> str:
        """Создает дата сэт - набор слов - следущее слово"""


class TextDataPreparation(iDataPreparation):
    LOCAL_DEVELOP_LINE_LIMIT = cfg.local_develop_line_limit

    def __init__(self, recreate_data=False):
        self._tokens_service: iTokensPreparation = get_tokens_service()
        self._recreate_data: bool = recreate_data

    def clean(self, text):
        text = text.lower()
        text = re.sub(r"@\w+|https?://\S+", "", text)
        text = re.sub(r"[^a-z0-9\s]", "", text)
        text = re.sub(r"([bcdfghjklmnpqrstvwxyz])\1{2,}", r"\1\1", text)
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
            yield self.clean(" ".join(current))

    def _batched_blocks(
        self, source: TextIO, batch_size: int, min_text_length: int
    ) -> Iterator[list[str]]:
        batch = []
        for block in self._iter_blocks(source):
            if len(block.split(" ")) < min_text_length:
                continue
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
        if not input_path.is_file():
            raise ValueError("Отсутствуют исходные данные")

        if not self._recreate_data and output_path.is_file():
            return "Документ уже существует, шаг пропущен"

        batch_count = 1

        with (
            input_path.open("r", encoding="utf-8") as input_file,
            output_path.open("w", encoding="utf-8", newline="") as output_file,
        ):
            for batch in tqdm(
                self._batched_blocks(input_file, batch_size, min_text_length),
                desc="Prepare raw csv",
                unit="batch",
            ):
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

    def _to_xy_pair(self, text: str) -> list[tuple[list[int], int]] | None:
        tokens = self._tokens_service.get_tokens(text)
        if len(tokens) <= DataConstants.X_length:
            return None

        pairs: list[tuple[list[int], int]] = []
        for i in range(len(tokens) - DataConstants.X_length):
            x_tokens = tokens[i : i + DataConstants.X_length]
            y_token = tokens[i + DataConstants.X_length]
            pairs.append((x_tokens, y_token))

        return pairs

    def create_next_token_dataset(
        self, input_path: Path, output_path: Path, batch_size: int = 1000
    ) -> str:
        if not input_path.is_file():
            raise ValueError("Отсутствуют входные данные для X/Y")

        if not self._recreate_data and output_path.is_file():
            return "Файл X/Y уже существует, шаг пропущен"

        total_rows = 0
        first_write = True

        with output_path.open("w", encoding="utf-8", newline="") as output_file:
            for chunk in tqdm(
                pd.read_csv(
                    input_path, names=["text"], header=None, chunksize=batch_size
                ),
                desc="Build X/Y dataset",
                unit="chunk",
            ):
                pairs: list[tuple[list[int], int]] = []
                for text in chunk["text"].dropna().astype(str):
                    text_pairs = self._to_xy_pair(text)
                    if text_pairs is not None:
                        pairs.extend(text_pairs)

                if not pairs:
                    continue

                df = pd.DataFrame(pairs, columns=["X", "Y"])
                df.to_csv(output_file, header=first_write, index=False)
                first_write = False
                total_rows += len(df)

        return f"Создан датасет X->Y, количество строк - {total_rows}"

    def split_train_val_test(
        self,
        input_path: Path,
        train_output_path: Path,
        val_output_path: Path,
        test_output_path: Path,
        seed: int = 42,
    ) -> str:
        if not input_path.is_file():
            raise ValueError("Отсутствуют данные для разделения train/val/test")

        existing_files = [
            path
            for path in [train_output_path, val_output_path, test_output_path]
            if path.is_file()
        ]
        if not self._recreate_data and existing_files:
            return (
                "Один или несколько файлов train/val/test уже существуют, шаг пропущен"
            )

        df = pd.read_csv(input_path)

        train_df, temp_df = train_test_split(
            df, test_size=0.2, random_state=seed, shuffle=True
        )
        val_df, test_df = train_test_split(
            temp_df, test_size=0.5, random_state=seed, shuffle=True
        )

        train_df.to_csv(train_output_path, index=False)
        val_df.to_csv(val_output_path, index=False)
        test_df.to_csv(test_output_path, index=False)

        train_count = len(train_df)
        val_count = len(val_df)
        test_count = len(test_df)

        total = train_count + val_count + test_count
        return (
            "Разделение завершено: "
            f"train={train_count}, val={val_count}, test={test_count}, total={total}"
        )
