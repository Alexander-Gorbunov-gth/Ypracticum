from dataclasses import dataclass
from logging import getLogger

from transformers import pipeline

from src.services.rouge_service import iTextMetricService
from src.services.tokens import iTokensPreparation
from src.services.training_dataset import NextTokenDataset

logger = getLogger(__name__)


@dataclass(frozen=True)
class BaselineMetrics:
    max_new_tokens: int
    do_sample: bool
    top_k: int
    rouge1: float
    rougeL: float
    eval_samples: int
    examples: list[tuple[str, str, str]]


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

    def run(
        self,
        val_dataset: NextTokenDataset,
        max_eval_samples: int = 500,
        examples_count: int = 8,
    ) -> BaselineMetrics:
        total_rouge1 = 0.0
        total_rougeL = 0.0
        samples = 0

        for index in range(min(len(val_dataset), max_eval_samples)):
            x_tokens, y_token = val_dataset[index]
            prompt_text = self._tokens_service.decode_tokens(x_tokens.tolist()).strip()
            target_text = self._tokens_service.decode_tokens([int(y_token.item())]).strip()
            if not prompt_text:
                continue

            pred_text = self._generate(prompt_text)
            rouge1, rougeL = self._metric_service.score_text(target_text, pred_text)

            total_rouge1 += rouge1
            total_rougeL += rougeL
            samples += 1

        examples = self._collect_examples(val_dataset, examples_count=examples_count)
        metrics = BaselineMetrics(
            max_new_tokens=5,
            do_sample=True,
            top_k=50,
            rouge1=total_rouge1 / max(samples, 1),
            rougeL=total_rougeL / max(samples, 1),
            eval_samples=samples,
            examples=examples,
        )

        print(
            f"\nDistilGPT2 baseline (val): rouge1={metrics.rouge1:.4f}, rougeL={metrics.rougeL:.4f}"
        )
        self.print_examples(metrics.examples)
        return metrics

    def _generate(self, prompt_text: str) -> str:
        result = self._generator(
            prompt_text,
            max_new_tokens=5,
            do_sample=True,
            top_k=50,
            pad_token_id=self._generator.tokenizer.eos_token_id,
            num_return_sequences=1,
            return_full_text=True,
        )
        generated_text = result[0]["generated_text"]
        completion = generated_text[len(prompt_text) :].strip()
        return completion.split(" ")[0].strip() if completion else ""

    def _collect_examples(
        self, dataset: NextTokenDataset, examples_count: int = 8
    ) -> list[tuple[str, str, str]]:
        examples: list[tuple[str, str, str]] = []
        for index in range(min(len(dataset), examples_count)):
            x_tokens, y_token = dataset[index]
            prompt_text = self._tokens_service.decode_tokens(x_tokens.tolist()).strip()
            target_text = self._tokens_service.decode_tokens([int(y_token.item())]).strip()
            pred_text = self._generate(prompt_text)
            examples.append((prompt_text, target_text, pred_text))
        return examples

    def print_examples(self, examples: list[tuple[str, str, str]]) -> None:
        print("\nПримеры DistilGPT2 baseline:")
        for index, (prompt_text, target_text, pred_text) in enumerate(examples, start=1):
            print(f"[{index}] input:  {prompt_text}")
            print(f"    target: {target_text}")
            print(f"    pred:   {pred_text}")
