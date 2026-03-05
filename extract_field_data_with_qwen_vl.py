from pathlib import Path

from extract_fields import ExtractedFields
from prompt_settings import ERROR_TOKEN
from ocr_client import ResponseExtractPaidEduContractDateFromImage, extract_paid_edu_contract_date_from_image_with_qwen3_vl
import logging

DIRECTORY_WITH_IMAGES = Path("agreements")
LOGGER = logging.getLogger(__name__)

def extract_filed_data_with_qwen_vl(stem: str, extracted_fields: ExtractedFields) -> ExctractedFields:
    """
    Если не удалось с помощью ocr достать поле data, пробуем еще раз с помощью qwen_vl
    """
    if extracted_fields.paid_edu_contract_date != ERROR_TOKEN:
        LOGGER.info("Дата была найдена ocr, поэтому vl не запускаем.")
        return extracted_fields

    if (extracted_fields.university_name != ERROR_TOKEN or extracted_fields.student_fio != ERROR_TOKEN or extracted_fields.customer_fio != ERROR_TOKEN or extracted_fields.paid_edu_contract_number != ERROR_TOKEN or extracted_fields.specialty_code != ERROR_TOKEN) and extracted_fields.paid_edu_contract_date == ERROR_TOKEN:
        image_path = DIRECTORY_WITH_IMAGES / f"{stem}.jpg"
        LOGGER.info("Дата не была найдена ocr, делаем запрос модельке qwen_vl, чтобы достать дату.")
        response_vl: ResponseExtractPaidEduContractDateFromImage = extract_paid_edu_contract_date_from_image_with_qwen3_vl(image_path)
        LOGGER.info("Получили ответ от qwen_vl.")
        if response_vl.success == False:
            LOGGER.error("Дата не была найдена. Ошибка извлечения.")
            return extracted_fields
        

        extracted_fields.paid_edu_contract_date = response_vl.date
        LOGGER.info("Достали дату из договра, заменили в extracted_fields.")
        return extracted_fields

    LOGGER.info("Выполнение запроса модели не потребовалось, скорее всего была ошибка изъятия текста договора OCR, возможно следует этот случай тоже обрабатывать.")
    return extracted_fields