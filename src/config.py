from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field
import logging
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
BASE_DIR = Path().resolve()


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=(".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )
    local_develop: bool = Field(default=True, validation_alias="LOCAL_DEVELOP")
    local_develop_line_limit: int = Field(
        default=30, validation_alias="LOCAL_DEVELOP_LINE_LIMIT"
    )
    training_batch_size: int = Field(default=256, validation_alias="TRAINING_BATCH_SIZE")
    training_epochs: int = Field(default=8, validation_alias="TRAINING_EPOCHS")
    training_search_epochs: int = Field(
        default=1, validation_alias="TRAINING_SEARCH_EPOCHS"
    )
    training_prompt_fraction: float = Field(
        default=0.75, validation_alias="TRAINING_PROMPT_FRACTION"
    )
    training_weight_decay: float = Field(
        default=0.0, validation_alias="TRAINING_WEIGHT_DECAY"
    )
    training_num_examples: int = Field(
        default=8, validation_alias="TRAINING_NUM_EXAMPLES"
    )
    training_max_new_tokens: int = Field(
        default=5, validation_alias="TRAINING_MAX_NEW_TOKENS"
    )
    lstm_embedding_dim: int = Field(default=128, validation_alias="LSTM_EMBEDDING_DIM")
    lstm_hidden_dim: int = Field(default=256, validation_alias="LSTM_HIDDEN_DIM")
    lstm_num_layers: int = Field(default=1, validation_alias="LSTM_NUM_LAYERS")
    lstm_dropout: float = Field(default=0.1, validation_alias="LSTM_DROPOUT")
    lstm_learning_rate: float = Field(
        default=2e-3, validation_alias="LSTM_LEARNING_RATE"
    )


cfg = Settings()
