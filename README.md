# Agreements DSPy Pipeline

## Как теперь работает извлечение

- LLM для DSPy конфигурируется один раз на запуск в `main.py`.
- Если найден `train_data.jsonl` (или задан `DSPY_TRAIN_JSONL`), extractor компилируется через `BootstrapFewShot`.
- Если trainset отсутствует или пустой, используется базовый extractor без компиляции.

## Переменные окружения

- `OPEN_ROUTER_API_KEY` (обязательная)
- `DSPY_MODEL` (опционально, по умолчанию `openrouter/qwen/qwen3-30b-a3b`)
- `DSPY_TIMEOUT_SEC` (таймаут одного запроса DSPy/LLM, по умолчанию `90`)
- `DSPY_NUM_RETRIES` (число ретраев DSPy/LLM, по умолчанию `1`)
- `DSPY_MAX_TOKENS` (лимит генерации DSPy/LLM, по умолчанию `512`)
- `OCR_BACKEND` (`qwen_vl` или `service`; по умолчанию `qwen_vl`)
- `OCR_VL_TIMEOUT_SEC` (таймаут одного OCR-запроса в Qwen-VL, по умолчанию `90`)
- `OCR_VL_MAX_RETRIES` (число ретраев OCR в Qwen-VL, по умолчанию `0`)
- `OCR_OPENAI_BASE_URL` (OpenAI-compatible URL для OCR-модели)
- `OCR_OPENAI_API_KEY` (ключ для OCR OpenAI-compatible endpoint, если требуется)
- `OCR_API_URL` (URL HTTP OCR сервиса для `OCR_BACKEND=service`)
- `OCR_HTTP_TIMEOUT_SEC` (таймаут HTTP OCR-сервиса, по умолчанию `120`)
- `DSPY_TRAIN_JSONL` (опционально, путь к JSONL с размеченными примерами)
- `DSPY_ONLINE_LEARNING` (`1/true` чтобы после каждого вызова добавлять пример и перекомпилировать)
- `DSPY_ONLINE_TRAIN_JSONL` (куда накапливать online-примеры, по умолчанию `online_train_data.jsonl`)
- `DSPY_ONLINE_RECOMPILE_EVERY` (через сколько принятых примеров перекомпилировать, по умолчанию `1`)
- `DSPY_ONLINE_MIN_NON_ERROR_FIELDS` (фильтр качества, минимум корректных полей из 6, по умолчанию `5`)
- `DSPY_ONLINE_MAX_SAMPLES` (лимит размера online-датасета, по умолчанию `300`)

## Формат train_data.jsonl

Каждая строка — отдельный JSON-объект:

```json
{
  "text": "OCR-текст договора ...",
  "university_name": "ФГАОУ ВО ...",
  "student_fio": "Иванов Иван Иванович",
  "customer_fio": "Петров Петр Петрович",
  "paid_edu_contract_number": "123/45",
  "paid_edu_contract_date": "2025-07-01",
  "specialty_code": "09.03.03",
  "evidence_json": "{\"student_fio\":\"...\"}"
}
```
