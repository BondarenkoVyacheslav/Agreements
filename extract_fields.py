from pydantic import BaseModel, Field
import dspy
from dspy.teleprompt import BootstrapFewShot
from typing import Optional, List
from ocr_client import extract_text_from_image_with_qwen3_vl, extract_text_from_image, OcrClientResponse
from utils import metric


class ExtractedFields(BaseModel):
    university_name: str = Field(description="Полное наименование ВУЗа/Исполнителя")
    student_fio: str = Field(description="ФИО обучающегося (полностью) или 'ОШИБКА'")
    customer_fio: str = Field(description="ФИО заказчика (полностью) или 'ОШИБКА'")
    paid_edu_contract_number: str = Field(description="Номер договора об оказании платных образовательных услуг или 'ОШИБКА'")
    paid_edu_contract_date: str = Field(description="Дата заключения договора (YYYY-MM-DD) или 'ОШИБКА'")
    specialty_code: str = Field(description="Код направления подготовки (например 09.03.03) или 'ОШИБКА'")

    # полезно для отладки/аудита:
    evidence: Optional[dict] = Field(default=None, description="Короткие цитаты/фрагменты, где найдено поле")


class ContractExtraction(dspy.Signature):
    """Извлеки ключевые поля из текста договора об образовании.

    Важно: ФИО обучающегося и заказчика могут быть написаны рукописным текстом.
    Такие фрагменты тоже нужно распознавать и извлекать.
    """
    text: str = dspy.InputField()
    university_name: str = dspy.OutputField()
    student_fio: str = dspy.OutputField(desc="ФИО обучающегося; может быть рукописным")
    customer_fio: str = dspy.OutputField(desc="ФИО заказчика; может быть рукописным")
    paid_edu_contract_number: str = dspy.OutputField()
    paid_edu_contract_date: str = dspy.OutputField(desc="YYYY-MM-DD или ОШИБКА")
    specialty_code: str = dspy.OutputField()


class Extractor(dspy.Module):
    def __init__(self):
        super().__init__()
        self.predict = dspy.Predict(ContractExtraction)

    def forward(self, text: str):
        return self.predict(text=text)


def build_trainset(rows: List[dict]) -> List[dspy.Example]:
    trainset = []
    for r in rows:
        trainset.append(
            dspy.Example(
                text=r["text"],
                university_name=r["university_name"],
                student_fio=r["student_fio"],
                customer_fio=r["customer_fio"],
                paid_edu_contract_number=r["paid_edu_contract_number"],
                paid_edu_contract_date=r["paid_edu_contract_date"],
                specialty_code=r["specialty_code"],
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


def extract_fields(compiled_extractor: dspy.Module, file_name: str) -> ExtractedFields:
    """
        Будем тут вызывать ocr_client.py, доставать текст.
        Доставать необходимые поля, дообучать программу извлечения,
    """
    ocr_client_response: OcrClientResponse = extract_text_from_image_with_qwen3_vl(file_name=file_name)

    if not ocr_client_response.success or not ocr_client_response.text:
        print("Ошибка извлечения текста!")
        return ExtractedFields(
            university_name="ОШИБКА",
            student_fio="ОШИБКА",
            customer_fio="ОШИБКА",
            paid_edu_contract_number="ОШИБКА",
            paid_edu_contract_date="ОШИБКА",
            specialty_code="ОШИБКА",
            evidence=None,
        )

    text = ocr_client_response.text

    pred = compiled_extractor(text=text)

    return ExtractedFields(
        university_name=getattr(pred, "university_name", "ОШИБКА") or "ОШИБКА",
        student_fio=getattr(pred, "student_fio", "ОШИБКА") or "ОШИБКА",
        customer_fio=getattr(pred, "customer_fio", "ОШИБКА") or "ОШИБКА",
        paid_edu_contract_number=getattr(pred, "paid_edu_contract_number", "ОШИБКА") or "ОШИБКА",
        paid_edu_contract_date=getattr(pred, "paid_edu_contract_date", "ОШИБКА") or "ОШИБКА",
        specialty_code=getattr(pred, "specialty_code", "ОШИБКА") or "ОШИБКА",
        evidence=None
    )




if __name__ == "__main__":
    import os
    import json
    import argparse

    # 1) Настраиваем LLM для DSPy (один раз)
    lm = dspy.LM("openrouter/qwen/qwen3-30b-a3b", api_key=os.environ["OPEN_ROUTER_API_KEY"])
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
