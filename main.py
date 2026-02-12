from pathlib import Path
import os
import json

import dspy

from transfer_excel_data import transfer_excel_data, transfer_reporting_data_to_excel, transfer_extracted_data_and_logic_to_excel
from extract_fields import build_trainset, compile_extractor, extract_fields, ExtractedFields, Extractor

DIRECTORY = Path("test_images")
TRAIN_JSONL: Path | None = None  # например: Path("train_data.jsonl")


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

    # Заполняем выгрузку данными которые нам передали
    transfer_excel_data()


    # Извлекаем все необходимые поля
    # Настраиваем LLM для DSPy (один раз)
    lm = dspy.LM("openrouter/qwen/qwen3-30b-a3b", api_key=os.environ["OPEN_ROUTER_API_KEY"])
    dspy.configure(lm=lm)

    compiled_extractor = build_compiled_extractor(TRAIN_JSONL)

    if not DIRECTORY.exists():
        raise FileNotFoundError(f"Images directory not found: {DIRECTORY}")

    images = [*DIRECTORY.glob("*.jpg")]
    if not images:
        raise FileNotFoundError(f"No images found in: {DIRECTORY}")

    for image in images:
        print(f"----{image}----")

        stem = image.stem  # "2025-06-05 12345"
        educational_loan_agreement_date, educational_loan_agreement_number = stem.replace("_", " ").split(maxsplit=1)

        result: ExtractedFields = extract_fields(compiled_extractor, image)

        # Заполняем поле 9 согласно дате заключения договора
        reporting_dates: str | None = None
        month: str | None = None 
        if result.paid_edu_contract_date != "ОШИБКА": 
            month = result.paid_edu_contract_date.split("-")[1] 

        if month == "07":
            reporting_dates = "ИЮЛЬ;АВГУСТ;СЕНТЯБРЬ"
        elif month == "08":
            reporting_dates = "АВГУСТ;СЕНТЯБРЬ"
        elif month == "09":
            reporting_dates = "СЕНТЯБРЬ"
        else:
            print("Проблема с месяцем заключения договора!")
        
        if not transfer_reporting_data_to_excel(educational_loan_agreement_number, educational_loan_agreement_date, reporting_dates):
            print(f"Договор с номером образовательного кредита:{educational_loan_agreement_number} и датой заключения кредита: {educational_loan_agreement_date} не был найден!")



        # Далее заполнямем извлеченные поля в соответствующие колоники
        if not transfer_extracted_data_and_logic_to_excel(educational_loan_agreement_number, educational_loan_agreement_date, result):
            print(f"Проблемы с заполнением полей от LLM или логики вывода, для договора с номером образовательного кредита:{educational_loan_agreement_number} и датой заключения кредита: {educational_loan_agreement_date}")


    return 0


if __name__ == "__main__":
    raise SystemExit(main())
    
