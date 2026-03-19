import ast
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset


class NextTokenDataset(Dataset):
    def __init__(self, input_path: Path) -> None:
        df = pd.read_csv(input_path)
        self._samples: list[tuple[list[int], int]] = []

        for _, row in df.iterrows():
            x_tokens = ast.literal_eval(row["X"])
            y_token = int(row["Y"])
            self._samples.append((x_tokens, y_token))

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        x_tokens, y_token = self._samples[index]
        x_tensor = torch.tensor(x_tokens, dtype=torch.long)
        y_tensor = torch.tensor(y_token, dtype=torch.long)
        return x_tensor, y_tensor


class DataLoaderFactory:
    def create(
        self, dataset: NextTokenDataset, batch_size: int, shuffle: bool
    ) -> DataLoader:
        return DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle)
