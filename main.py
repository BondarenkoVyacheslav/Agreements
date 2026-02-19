from pathlib import Path
import os
import json
import logging
from datetime import datetime

import dspy

from transfer_excel_data import transfer_excel_data, transfer_reporting_data_to_excel, transfer_extracted_data_and_logic_to_excel
from extract_fields import build_trainset, compile_extractor, extract_fields_from_text, ExtractedFields, Extractor

DIRECTORY = Path("extracted_texts")
TRAIN_JSONL: Path | None = None  # например: Path("train_data.jsonl")
MAX_FILES_TO_PROCESS = 6
LOGGER = logging.getLogger(__name__)
REPORTING_DATES_BY_MONTH = {
    "07": "ИЮЛЬ;АВГУСТ;СЕНТЯБРЬ",
    "08": "АВГУСТ;СЕНТЯБРЬ",
    "09": "СЕНТЯБРЬ",
}


def parse_contract_month(contract_date: str | None) -> str | None:
    if not contract_date:
        return None

    value = contract_date.strip()
    if not value or value.upper() == "ОШИБКА":
        return None

    formats = (
        "%Y-%m-%d",
        "%d.%m.%Y",
        "%d-%m-%Y",
        "%Y.%m.%d",
        "%d/%m/%Y",
        "%Y/%m/%d",
    )

    for date_value in (value, value.split()[0]):
        for date_format in formats:
            try:
                return datetime.strptime(date_value, date_format).strftime("%m")
            except ValueError:
                continue

    return None


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
    lm = dspy.LM("openrouter/qwen/qwen3-30b-a3b", api_key=os.environ["OPEN_ROUTER_API_KEY"])
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
            stem = text.stem  # "2025-06-05 12345"
            stem_parts = stem.replace("_", " ").split(maxsplit=1)
            if len(stem_parts) < 2:
                LOGGER.error("Пропуск файла с невалидным именем: %s", text.name)
                continue
            educational_loan_agreement_date, educational_loan_agreement_number = stem_parts

            result: ExtractedFields = extract_fields_from_text(compiled_extractor, text)
            LOGGER.info(
                "Поля извлечены для договора %s от %s",
                educational_loan_agreement_number,
                educational_loan_agreement_date,
            )
            LOGGER.debug("Извлечённые данные: %s", result)

            # Заполняем поле 9 согласно дате заключения договора
            month = parse_contract_month(result.paid_edu_contract_date)
            reporting_dates = REPORTING_DATES_BY_MONTH.get(month)
            if month is None:
                LOGGER.warning(
                    "Не удалось определить месяц по paid_edu_contract_date=%s",
                    result.paid_edu_contract_date,
                )
            elif reporting_dates is None:
                LOGGER.warning(
                    "Месяц %s вне ожидаемого диапазона (07-09), paid_edu_contract_date=%s",
                    month,
                    result.paid_edu_contract_date,
                )

            if not transfer_reporting_data_to_excel(educational_loan_agreement_number, educational_loan_agreement_date, reporting_dates):
                LOGGER.error(
                    "Договор с номером образовательного кредита %s и датой %s не найден при заполнении reporting-данных.",
                    educational_loan_agreement_number,
                    educational_loan_agreement_date,
                )
            LOGGER.info("Определены reporting_dates: %s", reporting_dates)

            # Далее заполнямем извлеченные поля в соответствующие колоники
            if not transfer_extracted_data_and_logic_to_excel(educational_loan_agreement_number, educational_loan_agreement_date, result):
                LOGGER.error(
                    "Проблемы с заполнением полей от LLM/логики вывода для договора %s от %s.",
                    educational_loan_agreement_number,
                    educational_loan_agreement_date,
                )
            LOGGER.info(
                "Заполнены извлечённые поля и сформирован вывод ИИ для договора %s от %s.",
                educational_loan_agreement_number,
                educational_loan_agreement_date,
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
    
