from abc import ABC, abstractmethod

from rouge_score import rouge_scorer

from src.services.tokens import iTokensPreparation, get_tokens_service


class iTextMetricService(ABC):
    @abstractmethod
    def score(
        self, target_tokens: list[int], pred_tokens: list[int]
    ) -> tuple[float, float]:
        """Возвращает rouge1 и rouge2."""

    @abstractmethod
    def score_text(self, target_text: str, pred_text: str) -> tuple[float, float]:
        """Возвращает rouge1 и rouge2 для строк."""


class RougeMetricService(iTextMetricService):
    def __init__(self) -> None:
        self._tokens_service: iTokensPreparation = get_tokens_service()
        self._scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2"], use_stemmer=True)

    def score(
        self, target_tokens: list[int], pred_tokens: list[int]
    ) -> tuple[float, float]:
        target_text = (
            self._tokens_service.decode_tokens(target_tokens).strip() or "[EMPTY]"
        )
        pred_text = self._tokens_service.decode_tokens(pred_tokens).strip() or "[EMPTY]"
        return self.score_text(target_text, pred_text)

    def score_text(self, target_text: str, pred_text: str) -> tuple[float, float]:
        target_text = target_text.strip() or "[EMPTY]"
        pred_text = pred_text.strip() or "[EMPTY]"
        scores = self._scorer.score(target_text, pred_text)
        return scores["rouge1"].fmeasure, scores["rouge2"].fmeasure


def get_metric_service() -> iTextMetricService:
    return RougeMetricService()
