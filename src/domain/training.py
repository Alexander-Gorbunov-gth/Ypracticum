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


@dataclass(frozen=True)
class EpochMetrics:
    accuracy: float
    loss: float
    rouge1: float
    rougeL: float
