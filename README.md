# Sentiment140 Next-Token Project

## Описание проекта
Проект реализует полный ML-пайплайн для задачи автодополнения текста на базе датасета `sentiment140`:
1. Подготовка и очистка данных.
2. Формирование датасета next-token (`X -> Y`).
3. Обучение LSTM-модели и оценка качества.
4. Сравнение с предобученным baseline `distilgpt2` из `transformers.pipeline`.
5. Формирование итогового текстового отчета.

Точка входа: `solution_project.py` или `solution_project.ipynb`.

## Структура
- `src/domain` — доменные сущности и конфиги.
- `src/services` — бизнес-логика по этапам (подготовка, обучение, baseline, отчет).
- `data` — входные/промежуточные/сплитованные датасеты.
- `models` — сохраненные веса модели.
- `reports` — итоговые отчеты.

## Ответственность классов

### Domain layer
- `DataConstants` (`src/domain/constans.py`)
  - Пути к данным и константы датасета (`X_length`, `batch_size` и т.д.).
- `ModelHyperParams`, `TrainingConfig`, `EpochMetrics` (`src/domain/training.py`)
  - Параметры модели, параметры обучения и структура метрик эпохи.

### Service layer
- `TextDataPreparation` (`src/services/data_utils.py`)
  - Очистка текста (`lowercase`, удаление мусора, нормализация пробелов).
  - Построение `dataset_processed.csv`.
  - Формирование next-token датасета `X_Y_dataset_file.csv`.
  - Разбиение на `train/val/test`.

- `TokensPreparation` (`src/services/tokens.py`)
  - Токенизация/детокенизация текста (`bert-base-uncased`).
  - Предоставление размера словаря.

- `NextTokenLSTM` (`src/services/lstm.py`)
  - LSTM-модель для предсказания следующего токена.
  - `forward` для обучения/инференса.
  - `generate` для генерации нескольких следующих токенов.

- `NextTokenDataset`, `DataLoaderFactory` (`src/services/training_dataset.py`)
  - Обертка над `train/val/test` CSV в `torch.Dataset`.
  - Создание `DataLoader`.

- `RougeMetricService` (`src/services/rouge_service.py`)
  - Подсчет метрик `ROUGE-1` и `ROUGE-2`.
  - Поддержка сравнения как токенов, так и строк.

- `LSTMTrainerService` (`src/services/lstm_trainer.py`)
  - Цикл обучения LSTM.
  - Валидация и тест (loss, accuracy, ROUGE).
  - Логирование метрик по эпохам.
  - Сохранение лучших весов.
  - Сбор примеров предсказаний.

- `DistilGPT2BaselineService` (`src/services/transformer_baseline.py`)
  - Использование готовой модели `distilgpt2` через `pipeline("text-generation")`.
  - Валидация и тест baseline по ROUGE.
  - Сбор примеров предсказаний baseline.

- `FinalReportService` (`src/services/report_service.py`)
  - Сбор всех результатов в единый текстовый отчет.
  - Сравнение LSTM vs DistilGPT2 и формирование рекомендации.

## Оркестрация
`solution_project.py` выполняет orchestration:
1. `prepare_datasets(...)`
2. `run_training()` (LSTM + DistilGPT2 baseline)
3. `FinalReportService.create(...)`

## Как запустить
```bash
./venv/bin/python solution_project.py
```

## Артефакты после запуска
- Веса LSTM: `models/next_token_lstm.pt`
- Итоговый отчет: `reports/final_report.txt`
