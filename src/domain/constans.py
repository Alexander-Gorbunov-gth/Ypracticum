from dataclasses import dataclass


@dataclass(frozen=True)
class DataConstants:
    raw_data_patch: str = "data/tweets.txt"
    dataset_processed_patch: str = "data/dataset_processed.csv"
    batch_size: int = 50
    CLS_token: int = 101
    SEP_token: int = 102
