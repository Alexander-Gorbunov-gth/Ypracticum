from datetime import datetime
from logging import getLogger
from pathlib import Path

from src.services.lstm_trainer import TrainResult
from src.services.transformer_baseline import BaselineMetrics

logger = getLogger(__name__)


class FinalReportService:
    def __init__(self, report_output_path: Path) -> None:
        self._report_output_path = report_output_path

    def create(
        self,
        prepare_results: dict[str, str],
        train_result: TrainResult,
        baseline_metrics: BaselineMetrics,
    ) -> Path:
        self._report_output_path.parent.mkdir(parents=True, exist_ok=True)
        generated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        lines: list[str] = []
        lines.append("ИТОГОВЫЙ ОТЧЕТ ПО ПРОЕКТУ")
        lines.append(f"Дата: {generated_at}")
        lines.append("")
        lines.append("1. Подготовка данных")
        lines.append(f"- Process dataset: {prepare_results['process_dataset_result']}")
        lines.append(f"- X/Y dataset: {prepare_results['xy_dataset_result']}")
        lines.append(f"- Split train/val/test: {prepare_results['split_result']}")
        lines.append("")
        lines.append("2. LSTM")
        lines.append(
            f"- Параметры: emb={train_result.best_params.embedding_dim}, "
            f"hid={train_result.best_params.hidden_dim}, "
            f"layers={train_result.best_params.num_layers}, "
            f"dropout={train_result.best_params.dropout}, "
            f"lr={train_result.best_params.learning_rate}"
        )
        lines.append(f"- Лучшая эпоха: {train_result.best_epoch}")
        lines.append(
            f"- Test: loss={train_result.test_metrics.loss:.4f}, "
            f"accuracy={train_result.test_metrics.accuracy:.4f}, "
            f"rouge1={train_result.test_metrics.rouge1:.4f}, "
            f"rougeL={train_result.test_metrics.rougeL:.4f}"
        )
        lines.append("- История по эпохам:")
        for item in train_result.epoch_history:
            lines.append(
                f"  epoch={int(item['epoch'])} | train_loss={item['train_loss']:.4f} | "
                f"val_accuracy={item['val_accuracy']:.4f} | val_loss={item['val_loss']:.4f} | "
                f"val_rouge1={item['val_rouge1']:.4f} | val_rougeL={item['val_rougeL']:.4f}"
            )
        lines.append("- Примеры LSTM:")
        for idx, (prompt_text, target_text, pred_text) in enumerate(
            train_result.examples, start=1
        ):
            lines.append(f"  [{idx}] input:  {prompt_text}")
            lines.append(f"      target: {target_text}")
            lines.append(f"      pred:   {pred_text}")
        lines.append("")
        lines.append("3. Baseline DistilGPT2 (pretrained, transformers.pipeline)")
        lines.append(
            f"- Параметры генерации: max_new_tokens={baseline_metrics.max_new_tokens}, "
            f"do_sample={baseline_metrics.do_sample}, top_k={baseline_metrics.top_k}"
        )
        lines.append(
            f"- Validation: rouge1={baseline_metrics.rouge1:.4f}, "
            f"rougeL={baseline_metrics.rougeL:.4f}, "
            f"samples={baseline_metrics.eval_samples}"
        )
        lines.append("- Примеры DistilGPT2:")
        for idx, (prompt_text, target_text, pred_text) in enumerate(
            baseline_metrics.examples, start=1
        ):
            lines.append(f"  [{idx}] input:  {prompt_text}")
            lines.append(f"      target: {target_text}")
            lines.append(f"      pred:   {pred_text}")
        lines.append("")
        lines.append("4. Артефакты")
        lines.append(f"- LSTM model path: {train_result.best_model_path}")
        lines.append(f"- Report path: {self._report_output_path}")

        self._report_output_path.write_text("\n".join(lines), encoding="utf-8")
        logger.info("Отчет сохранен: %s", self._report_output_path)
        return self._report_output_path
