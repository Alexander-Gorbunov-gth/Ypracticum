from logging import getLogger
from pathlib import Path

from src.config import BASE_DIR, cfg
from src.domain.constans import DataConstants
from src.domain.training import TrainingConfig
from src.services.data_utils import TextDataPreparation
from src.services.lstm_trainer import LSTMTrainerService
from src.services.report_service import FinalReportService
from src.services.rouge_service import get_metric_service
from src.services.tokens import get_tokens_service
from src.services.transformer_baseline import DistilGPT2BaselineService
from src.services.training_dataset import DataLoaderFactory, NextTokenDataset

logger = getLogger(__name__)

raw_data_file = Path(BASE_DIR / DataConstants.raw_data_patch)
dataset_processed_file = Path(BASE_DIR / DataConstants.dataset_processed_patch)
xy_dataset_file = Path(BASE_DIR / DataConstants.X_Y_dataset_file)
train_output_path = Path(BASE_DIR / DataConstants.train_output_path)
val_output_path = Path(BASE_DIR / DataConstants.val_output_path)
test_output_path = Path(BASE_DIR / DataConstants.test_output_path)
model_output_path = Path(BASE_DIR / "models" / "next_token_lstm.pt")
report_output_path = Path(BASE_DIR / "reports" / "final_report.txt")


def prepare_datasets(recreate_data: bool = False) -> dict[str, str]:
    text_service = TextDataPreparation(recreate_data=recreate_data)

    result = text_service.create_process_dataset(
        input_path=raw_data_file,
        output_path=dataset_processed_file,
        batch_size=DataConstants.batch_size,
        min_text_length=DataConstants.X_length + 1,
    )
    process_dataset_result = result
    logger.info(result)

    result = text_service.create_next_token_dataset(
        input_path=dataset_processed_file,
        output_path=xy_dataset_file,
        batch_size=DataConstants.batch_size,
    )
    xy_dataset_result = result
    logger.info(result)

    result = text_service.split_train_val_test(
        input_path=xy_dataset_file,
        train_output_path=train_output_path,
        val_output_path=val_output_path,
        test_output_path=test_output_path,
    )
    split_result = result
    logger.info(result)
    return {
        "process_dataset_result": process_dataset_result,
        "xy_dataset_result": xy_dataset_result,
        "split_result": split_result,
    }


def get_config_snapshot() -> dict[str, str]:
    return {
        "LOCAL_DEVELOP": str(cfg.local_develop),
        "LOCAL_DEVELOP_LINE_LIMIT": str(cfg.local_develop_line_limit),
        "TRAINING_BATCH_SIZE": str(cfg.training_batch_size),
        "TRAINING_EPOCHS": str(cfg.training_epochs),
        "TRAINING_SEARCH_EPOCHS": str(cfg.training_search_epochs),
        "TRAINING_PROMPT_FRACTION": str(cfg.training_prompt_fraction),
        "TRAINING_WEIGHT_DECAY": str(cfg.training_weight_decay),
        "TRAINING_NUM_EXAMPLES": str(cfg.training_num_examples),
        "TRAINING_MAX_NEW_TOKENS": str(cfg.training_max_new_tokens),
        "LSTM_EMBEDDING_DIM": str(cfg.lstm_embedding_dim),
        "LSTM_HIDDEN_DIM": str(cfg.lstm_hidden_dim),
        "LSTM_NUM_LAYERS": str(cfg.lstm_num_layers),
        "LSTM_DROPOUT": str(cfg.lstm_dropout),
        "LSTM_LEARNING_RATE": str(cfg.lstm_learning_rate),
        "DATA_X_LENGTH": str(DataConstants.X_length),
        "DATA_BATCH_SIZE_PREP": str(DataConstants.batch_size),
    }


def run_training() -> tuple:
    tokens_service = get_tokens_service()
    metric_service = get_metric_service()

    config = TrainingConfig(
        batch_size=cfg.training_batch_size,
        epochs=cfg.training_epochs,
        search_epochs=cfg.training_search_epochs,
        prompt_fraction=cfg.training_prompt_fraction,
        weight_decay=cfg.training_weight_decay,
        num_examples=cfg.training_num_examples,
        max_new_tokens=cfg.training_max_new_tokens,
        embedding_dim=cfg.lstm_embedding_dim,
        hidden_dim=cfg.lstm_hidden_dim,
        num_layers=cfg.lstm_num_layers,
        dropout=cfg.lstm_dropout,
        learning_rate=cfg.lstm_learning_rate,
    )

    train_dataset = NextTokenDataset(train_output_path)
    val_dataset = NextTokenDataset(val_output_path)
    test_dataset = NextTokenDataset(test_output_path)

    dataloader_factory = DataLoaderFactory()
    train_loader = dataloader_factory.create(
        dataset=train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
    )
    val_loader = dataloader_factory.create(
        dataset=val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
    )
    test_loader = dataloader_factory.create(
        dataset=test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
    )

    trainer = LSTMTrainerService(
        tokens_service=tokens_service,
        metric_service=metric_service,
        config=config,
        model_output_path=model_output_path,
    )

    result = trainer.run(
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        test_dataset=test_dataset,
    )

    logger.info(
        "Лучшие параметры: emb=%s hid=%s layers=%s dropout=%s lr=%s",
        result.best_params.embedding_dim,
        result.best_params.hidden_dim,
        result.best_params.num_layers,
        result.best_params.dropout,
        result.best_params.learning_rate,
    )
    logger.info(
        "Test metrics: loss=%.4f rouge1=%.4f rouge2=%.4f",
        result.test_metrics.loss,
        result.test_metrics.rouge1,
        result.test_metrics.rouge2,
    )
    logger.info("Модель сохранена: %s", result.best_model_path)

    baseline_service = DistilGPT2BaselineService(
        tokens_service=tokens_service,
        metric_service=metric_service,
    )
    baseline_val_metrics, baseline_test_metrics = baseline_service.run(
        val_dataset=val_dataset,
        test_dataset=test_dataset,
        examples_count=config.num_examples,
        max_new_tokens=config.max_new_tokens,
    )
    logger.info(
        "DistilGPT2 baseline val: rouge1=%.4f rouge2=%.4f | test: rouge1=%.4f rouge2=%.4f",
        baseline_val_metrics.rouge1,
        baseline_val_metrics.rouge2,
        baseline_test_metrics.rouge1,
        baseline_test_metrics.rouge2,
    )
    return result, baseline_val_metrics, baseline_test_metrics


if __name__ == "__main__":
    config_snapshot = get_config_snapshot()
    prepare_results = prepare_datasets(recreate_data=True)
    train_result, baseline_val_result, baseline_test_result = run_training()
    report_service = FinalReportService(report_output_path=report_output_path)
    report_service.create(
        config_snapshot,
        prepare_results,
        train_result,
        baseline_val_result,
        baseline_test_result,
    )
