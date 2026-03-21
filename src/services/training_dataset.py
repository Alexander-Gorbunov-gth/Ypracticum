import ast
import math
import random
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader, IterableDataset, get_worker_info


class NextTokenDataset(IterableDataset):
    def __init__(
        self,
        input_path: Path,
        chunksize: int = 10_000,
        shuffle_buffer_size: int = 0,
    ) -> None:
        self._input_path = input_path
        self._chunksize = chunksize
        self._shuffle_buffer_size = shuffle_buffer_size
        self._rows_count: int | None = None

    def _get_rows_count(self) -> int:
        if self._rows_count is not None:
            return self._rows_count

        with self._input_path.open("r", encoding="utf-8") as source:
            rows_count = max(sum(1 for _ in source) - 1, 0)
        self._rows_count = rows_count
        return rows_count

    def estimate_num_batches(self, batch_size: int) -> int:
        return math.ceil(self._get_rows_count() / batch_size) if batch_size > 0 else 0

    def _iter_samples(self):
        worker_info = get_worker_info()
        worker_id = worker_info.id if worker_info is not None else 0
        workers_count = worker_info.num_workers if worker_info is not None else 1
        row_index = 0

        for chunk in pd.read_csv(
            self._input_path,
            usecols=["X", "Y"],
            converters={"X": ast.literal_eval, "Y": int},
            chunksize=self._chunksize,
        ):
            x_values = chunk["X"].tolist()
            y_values = chunk["Y"].tolist()

            for x_tokens, y_token in zip(x_values, y_values):
                if row_index % workers_count == worker_id:
                    yield (
                        torch.tensor(x_tokens, dtype=torch.long),
                        torch.tensor(y_token, dtype=torch.long),
                    )
                row_index += 1

    def __iter__(self):
        if self._shuffle_buffer_size <= 1:
            yield from self._iter_samples()
            return

        buffer: list[tuple[torch.Tensor, torch.Tensor]] = []
        for sample in self._iter_samples():
            buffer.append(sample)
            if len(buffer) >= self._shuffle_buffer_size:
                idx = random.randrange(len(buffer))
                yield buffer.pop(idx)

        while buffer:
            idx = random.randrange(len(buffer))
            yield buffer.pop(idx)

    def __len__(self) -> int:
        return self._get_rows_count()


class DataLoaderFactory:
    def create(
        self, dataset: IterableDataset, batch_size: int, shuffle: bool
    ) -> DataLoader:
        shuffle = False
        return DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle)
