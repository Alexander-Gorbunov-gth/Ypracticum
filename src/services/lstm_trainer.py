from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader

from src.domain.training import EpochMetrics, ModelHyperParams, TrainingConfig
from src.services.lstm import NextTokenLSTM
from src.services.rouge_service import iTextMetricService
from src.services.tokens import iTokensPreparation
from src.services.training_dataset import NextTokenDataset


@dataclass(frozen=True)
class TrainResult:
    best_params: ModelHyperParams
    best_model_path: Path
    test_metrics: EpochMetrics


class LSTMTrainerService:
    def __init__(
        self,
        tokens_service: iTokensPreparation,
        metric_service: iTextMetricService,
        config: TrainingConfig,
        model_output_path: Path,
    ) -> None:
        self._tokens_service = tokens_service
        self._metric_service = metric_service
        self._config = config
        self._model_output_path = model_output_path
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def run(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: DataLoader,
        test_dataset: NextTokenDataset,
    ) -> TrainResult:
        params = ModelHyperParams(128, 256, 1, 0.1, 2e-3)

        model = self._build_model(params)
        criterion = nn.CrossEntropyLoss()
        optimizer = Adam(
            model.parameters(),
            lr=params.learning_rate,
            weight_decay=self._config.weight_decay,
        )

        best_val_loss = float("inf")
        print("\nФинальное обучение:")
        for epoch in range(1, self._config.epochs + 1):
            train_loss = self._train_one_epoch(
                model, train_loader, criterion, optimizer
            )
            val_metrics = self._evaluate(model, val_loader, criterion)

            print(
                f"Epoch {epoch:02d}/{self._config.epochs} | "
                f"train_loss={train_loss:.4f} | "
                f"val_loss={val_metrics.loss:.4f} | "
                f"val_rouge1={val_metrics.rouge1:.4f} | "
                f"val_rougeL={val_metrics.rougeL:.4f}"
            )

            if val_metrics.loss < best_val_loss:
                best_val_loss = val_metrics.loss
                self._model_output_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(model.state_dict(), self._model_output_path)

        model.load_state_dict(
            torch.load(self._model_output_path, map_location=self._device)
        )
        test_metrics = self._evaluate(model, test_loader, criterion)
        print(
            f"\nTest: loss={test_metrics.loss:.4f}, "
            f"rouge1={test_metrics.rouge1:.4f}, rougeL={test_metrics.rougeL:.4f}"
        )
        self.print_examples(model, test_dataset, num_examples=8)

        return TrainResult(
            best_params=params,
            best_model_path=self._model_output_path,
            test_metrics=test_metrics,
        )

    def _build_model(self, params: ModelHyperParams) -> NextTokenLSTM:
        return NextTokenLSTM(
            vocab_size=self._tokens_service.get_vocab_size(),
            embedding_dim=params.embedding_dim,
            hidden_dim=params.hidden_dim,
            num_layers=params.num_layers,
            dropout=params.dropout,
        ).to(self._device)

    def _get_prompt_batch(self, x_batch: torch.Tensor) -> torch.Tensor:
        prompt_len = max(1, int(x_batch.size(1) * self._config.prompt_fraction))
        return x_batch[:, :prompt_len]

    def _train_one_epoch(
        self,
        model: NextTokenLSTM,
        loader: DataLoader,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
    ) -> float:
        model.train()
        total_loss = 0.0

        for x_batch, y_batch in loader:
            x_batch = x_batch.to(self._device)
            y_batch = y_batch.to(self._device)
            prompt_batch = self._get_prompt_batch(x_batch)

            optimizer.zero_grad(set_to_none=True)
            logits, _ = model(prompt_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()

        return total_loss / len(loader)

    @torch.no_grad()
    def _evaluate(
        self, model: NextTokenLSTM, loader: DataLoader, criterion: nn.Module
    ) -> EpochMetrics:
        model.eval()
        total_loss = 0.0
        total_rouge1 = 0.0
        total_rougeL = 0.0
        samples = 0

        for x_batch, y_batch in loader:
            x_batch = x_batch.to(self._device)
            y_batch = y_batch.to(self._device)
            prompt_batch = self._get_prompt_batch(x_batch)

            logits, _ = model(prompt_batch)
            loss = criterion(logits, y_batch)
            total_loss += loss.item()

            generated = model.generate(
                input_ids=prompt_batch,
                max_new_tokens=1,
                do_sample=False,
            )
            pred_batch = generated[:, -1]

            for pred_token, target_token in zip(pred_batch, y_batch):
                rouge1, rougeL = self._metric_service.score(
                    target_tokens=[int(target_token.item())],
                    pred_tokens=[int(pred_token.item())],
                )
                total_rouge1 += rouge1
                total_rougeL += rougeL
                samples += 1

        return EpochMetrics(
            loss=total_loss / len(loader),
            rouge1=total_rouge1 / max(samples, 1),
            rougeL=total_rougeL / max(samples, 1),
        )

    @torch.no_grad()
    def print_examples(
        self,
        model: NextTokenLSTM,
        dataset: NextTokenDataset,
        num_examples: int = 8,
    ) -> None:
        model.eval()
        print("\nПримеры автодополнений (3/4 -> 1/4):")

        for idx in range(min(num_examples, len(dataset))):
            x_tokens, y_token = dataset[idx]

            x_batch = x_tokens.unsqueeze(0).to(self._device)
            prompt = self._get_prompt_batch(x_batch)

            generated = model.generate(
                input_ids=prompt,
                max_new_tokens=1,
                do_sample=False,
            )
            pred_token = int(generated[0, -1].item())
            target_token = int(y_token.item())

            prompt_text = self._tokens_service.decode_tokens(prompt[0].cpu().tolist())
            target_text = self._tokens_service.decode_tokens([target_token])
            pred_text = self._tokens_service.decode_tokens([pred_token])

            print(f"[{idx + 1}] input:  {prompt_text}")
            print(f"    target: {target_text}")
            print(f"    pred:   {pred_text}")
