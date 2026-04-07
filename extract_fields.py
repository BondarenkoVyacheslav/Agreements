import dspy
from dspy.teleprompt import BootstrapFewShot
from typing import Optional, List
from pathlib import Path
from utils import metric
import json
from pydantic import BaseModel, Field
from normalizes import _normalize_value, _normalize_date, _normalize_specialty, _normalize_evidence

from ocr_client import (
    build_local_text_lm,
    OcrClientResponse,
    extract_text_from_image,
    extract_text_from_image_with_qwen3_vl,
)
from prompt_settings import ERROR_TOKEN, TwoStageExtractor


class ExtractedFields(BaseModel):
    university_name: str = Field(description="Полное наименование ВУЗа/Исполнителя")
    student_fio: str = Field(description="ФИО обучающегося (полностью) или 'ОШИБКА'")
    customer_fio: str = Field(description="ФИО заказчика (полностью) или 'ОШИБКА'")
    paid_edu_contract_number: str = Field(description="Номер договора об оказании платных образовательных услуг или 'ОШИБКА'")
    paid_edu_contract_date: str = Field(description="Дата заключения договора (YYYY-MM-DD) или 'ОШИБКА'")
    specialty_code: str = Field(description="Код направления подготовки (например 09.03.03) или 'ОШИБКА'")

    # полезно для отладки/аудита:
    evidence: Optional[dict] = Field(default=None, description="Короткие фрагменты, где найдено поле")


class Extractor(TwoStageExtractor):
    """Backwards-compatible alias for main.py and compilation flow."""


def _row_evidence(row: dict) -> dict:
    if isinstance(row.get("evidence"), dict):
        return row["evidence"]

    evidence_json = row.get("evidence_json")
    if isinstance(evidence_json, str) and evidence_json.strip():
        try:
            parsed = json.loads(evidence_json)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass

    return {}


def _get_train_value(row: dict, key: str, fallback: str = ERROR_TOKEN) -> str:
    value = row.get(key, fallback)
    value = _normalize_value(value)
    return value if value else fallback


def build_trainset(rows: List[dict]) -> List[dspy.Example]:
    trainset = []
    for row in rows:
        evidence = _row_evidence(row)
        trainset.append(
            dspy.Example(
                text=row["text"],
                university_name=row["university_name"],
                student_fio=row["student_fio"],
                customer_fio=row["customer_fio"],
                paid_edu_contract_number=row["paid_edu_contract_number"],
                paid_edu_contract_date=row["paid_edu_contract_date"],
                specialty_code=row["specialty_code"],
                evidence_university_name=_get_train_value(
                    row,
                    "evidence_university_name",
                    _get_train_value(evidence, "university_name"),
                ),
                evidence_student_fio=_get_train_value(
                    row,
                    "evidence_student_fio",
                    _get_train_value(evidence, "student_fio"),
                ),
                evidence_customer_fio=_get_train_value(
                    row,
                    "evidence_customer_fio",
                    _get_train_value(evidence, "customer_fio"),
                ),
                evidence_paid_edu_contract_number=_get_train_value(
                    row,
                    "evidence_paid_edu_contract_number",
                    _get_train_value(evidence, "paid_edu_contract_number"),
                ),
                evidence_paid_edu_contract_date=_get_train_value(
                    row,
                    "evidence_paid_edu_contract_date",
                    _get_train_value(evidence, "paid_edu_contract_date"),
                ),
                evidence_specialty_code=_get_train_value(
                    row,
                    "evidence_specialty_code",
                    _get_train_value(evidence, "specialty_code"),
                ),
            ).with_inputs("text")
        )
    return trainset

def compile_extractor(trainset: List[dspy.Example]) -> dspy.Module:
    """Компиляция (оптимизация few-shot) делается один раз."""
    teleprompter = BootstrapFewShot(
        metric=metric,
        max_bootstrapped_demos=8,
        max_labeled_demos=8,
    )
    return teleprompter.compile(Extractor(), trainset=trainset)


def _empty_extracted_fields() -> ExtractedFields:
    return ExtractedFields(
        university_name=ERROR_TOKEN,
        student_fio=ERROR_TOKEN,
        customer_fio=ERROR_TOKEN,
        paid_edu_contract_number=ERROR_TOKEN,
        paid_edu_contract_date=ERROR_TOKEN,
        specialty_code=ERROR_TOKEN,
        evidence={
            "university_name": ERROR_TOKEN,
            "student_fio": ERROR_TOKEN,
            "customer_fio": ERROR_TOKEN,
            "paid_edu_contract_number": ERROR_TOKEN,
            "paid_edu_contract_date": ERROR_TOKEN,
            "specialty_code": ERROR_TOKEN,
        },
    )


def _build_extracted_fields_from_prediction(pred: object) -> ExtractedFields:
    university_name = _normalize_value(getattr(pred, "university_name", ERROR_TOKEN))
    student_fio = _normalize_value(getattr(pred, "student_fio", ERROR_TOKEN))
    customer_fio = _normalize_value(getattr(pred, "customer_fio", ERROR_TOKEN))
    paid_edu_contract_number = _normalize_value(getattr(pred, "paid_edu_contract_number", ERROR_TOKEN))
    paid_edu_contract_date = _normalize_date(getattr(pred, "paid_edu_contract_date", ERROR_TOKEN))
    specialty_code = _normalize_specialty(getattr(pred, "specialty_code", ERROR_TOKEN))

    evidence = {
        "university_name": _normalize_evidence(getattr(pred, "evidence_university_name", ERROR_TOKEN), university_name),
        "student_fio": _normalize_evidence(getattr(pred, "evidence_student_fio", ERROR_TOKEN), student_fio),
        "customer_fio": _normalize_evidence(getattr(pred, "evidence_customer_fio", ERROR_TOKEN), customer_fio),
        "paid_edu_contract_number": _normalize_evidence(
            getattr(pred, "evidence_paid_edu_contract_number", ERROR_TOKEN),
            paid_edu_contract_number,
        ),
        "paid_edu_contract_date": _normalize_evidence(
            getattr(pred, "evidence_paid_edu_contract_date", ERROR_TOKEN),
            paid_edu_contract_date,
        ),
        "specialty_code": _normalize_evidence(getattr(pred, "evidence_specialty_code", ERROR_TOKEN), specialty_code),
    }

    return ExtractedFields(
        university_name=university_name,
        student_fio=student_fio,
        customer_fio=customer_fio,
        paid_edu_contract_number=paid_edu_contract_number,
        paid_edu_contract_date=paid_edu_contract_date,
        specialty_code=specialty_code,
        evidence=evidence,
    )


def extract_fields_from_text_content(compiled_extractor: dspy.Module, text: str) -> ExtractedFields:
    if not text or not text.strip():
        return _empty_extracted_fields()

    pred = compiled_extractor(text=text.strip())
    return _build_extracted_fields_from_prediction(pred)


def extract_fields(compiled_extractor: dspy.Module, file_name: str) -> ExtractedFields:
    """
       Вариант с OCR: достаем текст из изображения и извлекаем необходимые поля.
    """
    ocr_client_response: OcrClientResponse = extract_text_from_image(file_name=file_name)

    if not ocr_client_response.success or not ocr_client_response.text:
        print("Ошибка извлечения текста!")
        return _empty_extracted_fields()

    return extract_fields_from_text_content(compiled_extractor, ocr_client_response.text)


def extract_fields_from_image_with_qwen3_vl_ocr(compiled_extractor: dspy.Module, file_name: str) -> ExtractedFields:
    ocr_client_response: OcrClientResponse = extract_text_from_image_with_qwen3_vl(file_name=file_name)

    if not ocr_client_response.success or not ocr_client_response.text:
        print("Ошибка извлечения текста через Qwen3-VL OCR!")
        return _empty_extracted_fields()

    return extract_fields_from_text_content(compiled_extractor, ocr_client_response.text)


def extract_fields_from_text(compiled_extractor: dspy.Module, file_name: str) -> ExtractedFields:
    """
        Сразу достаем из говтовго txt файла, необходимые поля.
    """
    source_name = Path(str(file_name)).name

    if source_name.lower().endswith(".txt"):
        text_file_path = Path(file_name)
        if not text_file_path.exists():
            text_file_path = Path("extracted_texts") / source_name
    else:
        stem = Path(source_name).stem
        text_file_path = Path("extracted_texts") / f"{stem}.txt"
        if not text_file_path.exists():
            text_file_path = Path("extracted_texts") / f"{stem.replace(' ', '_')}.txt"
        if not text_file_path.exists():
            text_file_path = Path("extracted_texts") / f"{stem.replace('_', ' ')}.txt"

    if not text_file_path.exists():
        print(f"Файл с извлеченным текстом не найден: {text_file_path}")
        return _empty_extracted_fields()

    with open(file=text_file_path, mode="r", encoding="utf-8") as f:
        text = f.read().strip()

    if text == "Processing failed: request timed out":
        print(f"Файл с извлеченным текстом пуст: {text_file_path}. Содержимое файла {text}: ")
        return _empty_extracted_fields()

    if not text:
        print(f"Файл с извлеченным текстом пуст: {text_file_path}")
        return _empty_extracted_fields()

    return extract_fields_from_text_content(compiled_extractor, text)

    



if __name__ == "__main__":
    import os
    import json
    import argparse

    # 1) Настраиваем LLM для DSPy (один раз)
    lm = build_local_text_lm()
    dspy.configure(lm=lm)

    # 2) CLI аргументы
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train",
        default=None,
        help="JSONL с размеченными примерами для компиляции (опционально)",
    )
    args = parser.parse_args()

    # Жестко заданный путь к входному файлу
    file_name = "test_images/01.02.2025 130291.jpg"

    # 3) Готовим extractor (скомпилированный или базовый)
    if args.train:
        rows = []
        with open(args.train, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rows.append(json.loads(line))
        trainset = build_trainset(rows)
        compiled_extractor = compile_extractor(trainset)
    else:
        compiled_extractor = Extractor()  # без компиляции (временно)

    # 4) Инференс
    result = extract_fields(compiled_extractor, file_name)
    print(result.model_dump_json(ensure_ascii=False, indent=2))
