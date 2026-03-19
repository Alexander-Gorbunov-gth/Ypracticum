from dataclasses import dataclass


@dataclass(frozen=True)
class DataConstants:
    raw_data_patch: str = "data/tweets.txt"
    dataset_processed_patch: str = "data/dataset_processed.csv"
    X_Y_dataset_file: str = "data/X_Y_dataset_file.csv"
    train_output_path: str = "data/train_output.csv"
    val_output_path: str = "data/val_output.csv"
    test_output_path: str = "data/test_output.csv"
    batch_size: int = 50
    CLS_token: int = 101
    SEP_token: int = 102
    X_length: int = 7
