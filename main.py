from pathlib import Path
import os
import json
import logging

import dspy

from transfer_excel_data import (
    extract_credit_contract_number_from_document_name,
    transfer_excel_data,
    transfer_extracted_data_and_logic_to_excel,
)
from extract_fields import build_trainset, compile_extractor, extract_fields_from_text, ExtractedFields, Extractor
from ocr_client import build_local_text_lm

DIRECTORY = Path("extracted_texts")
TRAIN_JSONL: Path | None = None  # например: Path("train_data.jsonl")
MAX_FILES_TO_PROCESS = 6
LOGGER = logging.getLogger(__name__)


def configure_logging() -> None:
    log_level_name = os.getenv("LOG_LEVEL", "INFO").upper()
    log_level = getattr(logging, log_level_name, logging.INFO)
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def load_rows(train_path: Path) -> list[dict]:
    rows: list[dict] = []
    with train_path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def build_compiled_extractor(train_path: Path | None) -> dspy.Module:
    if train_path is None:
        return Extractor()
    if not train_path.exists():
        raise FileNotFoundError(f"Train file not found: {train_path}")

    rows = load_rows(train_path)
    trainset = build_trainset(rows)
    return compile_extractor(trainset)


def main() -> int:
    """
    Тут собираем все модули и реализуем всю логику MVP
    
    :return: int
    :rtype: int
    """
    configure_logging()

    # Заполняем выгрузку данными которые нам передали
    transfer_excel_data()


    # Извлекаем все необходимые поля
    # Настраиваем LLM для DSPy (один раз)
    lm = build_local_text_lm()
    dspy.configure(lm=lm)

    compiled_extractor = build_compiled_extractor(TRAIN_JSONL)

    if not DIRECTORY.exists():
        raise FileNotFoundError(f"Images directory not found: {DIRECTORY}")

    all_texts = sorted(DIRECTORY.glob("*.txt"))
    if not all_texts:
        raise FileNotFoundError(f"No text files found in: {DIRECTORY}")
    LOGGER.info(
        "Найдено %d Текстовых файлов.",
        len(all_texts),
    )
    all_texts = all_texts[:MAX_FILES_TO_PROCESS]

    for i, text in enumerate(all_texts):
        LOGGER.info(f"Текстовый файл №{i}")
        LOGGER.info("Обрабатываем текстовый файл: %s", text)
        try:
            educational_loan_agreement_number = extract_credit_contract_number_from_document_name(text)
            if educational_loan_agreement_number is None:
                LOGGER.error("Пропуск файла с невалидным именем: %s", text.name)
                continue

            result: ExtractedFields = extract_fields_from_text(compiled_extractor, text)
            LOGGER.info(
                "Поля извлечены для договора %s",
                educational_loan_agreement_number,
            )
            LOGGER.debug("Извлечённые данные: %s", result)

            # Далее заполнямем извлеченные поля в соответствующие колоники
            if not transfer_extracted_data_and_logic_to_excel(educational_loan_agreement_number, result):
                LOGGER.error(
                    "Проблемы с заполнением полей от LLM/логики вывода для договора %s.",
                    educational_loan_agreement_number,
                )
            LOGGER.info(
                "Заполнены извлечённые поля и сформирован вывод ИИ для договора %s.",
                educational_loan_agreement_number,
            )
        except Exception:
            LOGGER.exception(
                "Необработанная ошибка при обработке изображения %s. Переходим к следующему.",
                text,
            )
            continue

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
    
