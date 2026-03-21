from logging import getLogger

from tqdm.auto import tqdm
from transformers import pipeline

from src.domain.training import BaselineMetrics
from src.services.rouge_service import iTextMetricService
from src.services.tokens import iTokensPreparation
from src.services.training_dataset import NextTokenDataset

logger = getLogger(__name__)


class DistilGPT2BaselineService:
    def __init__(
        self,
        tokens_service: iTokensPreparation,
        metric_service: iTextMetricService,
    ) -> None:
        self._tokens_service = tokens_service
        self._metric_service = metric_service
        self._generator = pipeline("text-generation", model="distilgpt2")
        if self._generator.tokenizer.pad_token_id is None:
            self._generator.tokenizer.pad_token = self._generator.tokenizer.eos_token
        logger.info("DistilGPT2 baseline инициализирован через transformers.pipeline")

    def _evaluate_dataset(
        self,
        dataset: NextTokenDataset,
        split: str,
        max_eval_samples: int,
        examples_count: int,
        max_new_tokens: int,
    ) -> BaselineMetrics:
        total_rouge1 = 0.0
        total_rouge2 = 0.0
        samples = 0

        for x_tokens, y_token in tqdm(
            dataset,
            total=max_eval_samples,
            desc=f"Baseline {split}",
            unit="sample",
        ):
            if samples >= max_eval_samples:
                break
            prompt_text = self._tokens_service.decode_tokens(x_tokens.tolist()).strip()
            target_text = self._tokens_service.decode_tokens(
                [int(y_token.item())]
            ).strip()
            if not prompt_text:
                continue

            pred_text = self._generate(prompt_text, max_new_tokens=max_new_tokens)
            rouge1, rouge2 = self._metric_service.score_text(target_text, pred_text)

            total_rouge1 += rouge1
            total_rouge2 += rouge2
            samples += 1

        examples = self._collect_examples(
            dataset,
            examples_count=examples_count,
            max_new_tokens=max_new_tokens,
        )
        metrics = BaselineMetrics(
            split=split,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            top_k=50,
            rouge1=total_rouge1 / max(samples, 1),
            rouge2=total_rouge2 / max(samples, 1),
            eval_samples=samples,
            examples=examples,
        )

        print(
            f"\nDistilGPT2 baseline ({split}): rouge1={metrics.rouge1:.4f}, rouge2={metrics.rouge2:.4f}"
        )
        self.print_examples(metrics.examples)
        return metrics

    def _generate(self, prompt_text: str, max_new_tokens: int) -> str:
        result = self._generator(
            prompt_text,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            top_k=50,
            pad_token_id=self._generator.tokenizer.eos_token_id,
            num_return_sequences=1,
            return_full_text=True,
        )
        generated_text = result[0]["generated_text"]
        completion = generated_text[len(prompt_text) :].strip()
        return completion

    def _collect_examples(
        self,
        dataset: NextTokenDataset,
        examples_count: int = 8,
        max_new_tokens: int = 1,
    ) -> list[tuple[str, str, str]]:
        examples: list[tuple[str, str, str]] = []
        for x_tokens, y_token in tqdm(
            dataset,
            total=examples_count,
            desc="Baseline examples",
            unit="sample",
            leave=False,
        ):
            if len(examples) >= examples_count:
                break
            prompt_text = self._tokens_service.decode_tokens(x_tokens.tolist()).strip()
            target_text = self._tokens_service.decode_tokens(
                [int(y_token.item())]
            ).strip()
            pred_text = self._generate(prompt_text, max_new_tokens=max_new_tokens)
            examples.append((prompt_text, target_text, pred_text))
        return examples

    def print_examples(self, examples: list[tuple[str, str, str]]) -> None:
        print("\nПримеры DistilGPT2 baseline:")
        for index, (prompt_text, target_text, pred_text) in enumerate(
            examples, start=1
        ):
            print(f"[{index}] input:  {prompt_text}")
            print(f"    target: {target_text}")
            print(f"    pred:   {pred_text}")

    def run(
        self,
        val_dataset: NextTokenDataset,
        test_dataset: NextTokenDataset,
        max_eval_samples: int = 500,
        examples_count: int = 8,
        max_new_tokens: int = 1,
    ) -> tuple[BaselineMetrics, BaselineMetrics]:
        val_metrics = self._evaluate_dataset(
            dataset=val_dataset,
            split="val",
            max_eval_samples=max_eval_samples,
            examples_count=examples_count,
            max_new_tokens=max_new_tokens,
        )
        test_metrics = self._evaluate_dataset(
            dataset=test_dataset,
            split="test",
            max_eval_samples=max_eval_samples,
            examples_count=examples_count,
            max_new_tokens=max_new_tokens,
        )
        return val_metrics, test_metrics
