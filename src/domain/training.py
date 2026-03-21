from dataclasses import dataclass


@dataclass(frozen=True)
class ModelHyperParams:
    embedding_dim: int
    hidden_dim: int
    num_layers: int
    dropout: float
    learning_rate: float


@dataclass(frozen=True)
class TrainingConfig:
    batch_size: int = 256
    epochs: int = 8
    search_epochs: int = 1
    prompt_fraction: float = 0.75
    weight_decay: float = 0.0
    num_examples: int = 8
    max_new_tokens: int = 5
    embedding_dim: int = 128
    hidden_dim: int = 256
    num_layers: int = 1
    dropout: float = 0.1
    learning_rate: float = 2e-3


@dataclass(frozen=True)
class EpochMetrics:
    accuracy: float
    loss: float
    rouge1: float
    rouge2: float


@dataclass(frozen=True)
class BaselineMetrics:
    split: str
    max_new_tokens: int
    do_sample: bool
    top_k: int
    rouge1: float
    rouge2: float
    eval_samples: int
    examples: list[tuple[str, str, str]]
