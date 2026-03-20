from logging import getLogger
from pathlib import Path

from src.config import BASE_DIR
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


def run_training() -> tuple:
    tokens_service = get_tokens_service()
    metric_service = get_metric_service()

    config = TrainingConfig(
        batch_size=256,
        epochs=8,
        search_epochs=1,
        prompt_fraction=0.75,
        weight_decay=0.0,
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
        "Test metrics: loss=%.4f rouge1=%.4f rougeL=%.4f",
        result.test_metrics.loss,
        result.test_metrics.rouge1,
        result.test_metrics.rougeL,
    )
    logger.info("Модель сохранена: %s", result.best_model_path)

    baseline_service = DistilGPT2BaselineService(
        tokens_service=tokens_service,
        metric_service=metric_service,
    )
    baseline_metrics = baseline_service.run(val_dataset=val_dataset)
    logger.info(
        "DistilGPT2 baseline metrics: rouge1=%.4f rougeL=%.4f",
        baseline_metrics.rouge1,
        baseline_metrics.rougeL,
    )
    return result, baseline_metrics


if __name__ == "__main__":
    prepare_results = prepare_datasets(recreate_data=True)
    train_result, baseline_result = run_training()
    report_service = FinalReportService(report_output_path=report_output_path)
    report_service.create(prepare_results, train_result, baseline_result)
