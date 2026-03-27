## LNG ML Research

Проект использует `json/jsonl` как единственный входной raw-формат.
CSV-входы не используются как источник для модели и STS-анализа.

Поддерживаемые входы в `data/raw`:
- `ais_observations.jsonl`
- `vessel_zone_events.jsonl`
- `sts_candidates.jsonl`
- `lng_tracker_dataset.json`

## Anomaly Detection

`LNGDataProcessor` собирает event-level датасет из `json/jsonl`, а `AnomalyModel`
ранжирует проходы судов по `risk_score`.

```python
from lng_ml_research.models import AnomalyModel
from lng_ml_research.processor import LNGDataProcessor

processor = LNGDataProcessor("data/raw")
raw_df = processor.load_data()
features_df = processor.prepare_features(raw_df)

model = AnomalyModel(contamination=0.1)
top_risky = model.top_anomalies(features_df, top_n=10)

print(top_risky)
```

Модель учитывает:
- зону прохода
- статус события
- время входа
- длительность события
- отклонение длительности от типичной по зоне

Итоговый `risk_score` объединяет `IsolationForest` и интерпретируемый сигнал,
показывающий насколько событие длиннее медианного для своей зоны.

## Notebook

В [notebooks/exploration.ipynb](/e:/funny/lng-ml-research/notebooks/exploration.ipynb)
ноутбук по умолчанию читает локальный `data/raw` и показывает доступные `json/jsonl`
файлы перед запуском пайплайна.
