import base64
import logging
import os
import re
from datetime import datetime
from pathlib import Path
from time import perf_counter, sleep
from dataclasses import dataclass

import requests
from openai import OpenAI
import dspy

LOGGER = logging.getLogger(__name__)

BASE_URL = "https://ocr.g-309.ru"
OPENAI_BASE_URL = "http://10.14.49.32:32204/v1"
OPENAI_API_KEY = ""
LOCAL_TEXT_LLM_MODEL = "openai/qwen3.5-35b-a3b-ud-q8_k_xl"
LOCAL_TEXT_LLM_MAX_TOKENS = 8192


def normalize_local_api_base(api_url: str) -> str:
    value = str(api_url).strip().rstrip("/")
    if not value:
        return value
    if value.endswith("/v1"):
        return value
    return f"{value}/v1"


def build_local_text_lm() -> dspy.LM:
    return dspy.LM(
        LOCAL_TEXT_LLM_MODEL,
        api_base=normalize_local_api_base(os.environ["API_URL"]),
        api_key=os.environ["API_KEY"],
        max_tokens=LOCAL_TEXT_LLM_MAX_TOKENS,
    )

@dataclass
class OcrClientResponse:
    text: str | None
    success: bool


def extract_text_from_image(file_name: str) -> OcrClientResponse:
    with open(file_name, "rb") as f:
        r = requests.post(f"{BASE_URL}/api/v1/ocr", files={"uploaded_file": f}, timeout=120)
    r.raise_for_status()
    payload = r.json()
    return OcrClientResponse(text=payload.get("text"), success=bool(payload.get("success")))



def extract_text_from_image_with_qwen3_vl(file_name: str) -> OcrClientResponse:
    timeout_sec = float(os.getenv("OCR_VL_TIMEOUT_SEC", "90"))
    max_retries = int(os.getenv("OCR_VL_MAX_RETRIES", "0"))

    client = OpenAI(
        base_url=OPENAI_BASE_URL,
        api_key=OPENAI_API_KEY,
        timeout=timeout_sec,
        max_retries=max_retries,
    )

    try:
        with open(file_name, "rb") as f:
            image_base64 = base64.b64encode(f.read()).decode("utf-8")

        response = client.chat.completions.create(
            model="Qwen/Qwen3-VL-4B-Instruct",
            timeout=timeout_sec,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": "Extract all text from this image. Return only the text."},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}},
                ],
            }],
        )
        text = response.choices[0].message.content
        text = text.strip() if isinstance(text, str) else None
        return OcrClientResponse(text=text, success=bool(text))
    except Exception as e:
        logging.getLogger(__name__).error(
            "Qwen3-VL OCR failed for %s (timeout=%ss, retries=%s): %s",
            file_name,
            timeout_sec,
            max_retries,
            e,
        )
        return OcrClientResponse(text=None, success=False)
    
def extract_fields_with_qwen_35b() -> OcrClientResponse:
    """Кидает запрос на сервер для извлечения полей из релевантных фрагментов с помощью модели Qwen3.5-35b-a3b-ud-q8_k_xl."""
    client = OpenAI(
        base_url=os.getenv("API_URL", ""),
        api_key=os.getenv("API_KEY", ""),
    )
    pass

    

# def extract_paid_edu_contract_date_from_image_with_ocr(file_name: str) -> ResponsExtractPaidEduContractDateFromImage:
#     """
#         Необходимо повторно попытаться достать дату заключения договора из изображения.
#     """
#     prompt = f"""
#         Необходимо досать дату договора, 
#         чаще всего дата договора находиться справа свехрху без подписи, что это дата договора,
#         в формате день: число в кавычках "" или << >>, 
#         месяц: может быть как числом так и словом,
#         год: четерех значаное число. 

#         Так же дата заключения договора может находиться и в самом документе, нужно досать именно дату заключения договора.
#         Бывают даты действия закона или каких то иных нормативных актов.

#         Результат строго дата в формате XX:XX:XXXX - день:месяц:год
#         """
#     with open(file_name, "rb") as f:
#         r = requests.post(f"{BASE_URL}/api/v1/ocr", files={"uploaded_file": f}, timeout=120)
#     r.raise_for_status()
#     payload = r.json()
#     return ResponsExtractPaidEduContractDateFromImage(text=payload.get("text"), success=bool(payload.get("success")))


CONTRACT_DATE_PROMPT = """
    Ты анализируешь изображение договора.
    Нужно извлечь именно дату заключения договора об оказании платных образовательных услуг.

    Правила:
    1) Игнорируй даты законов, лицензий, приказов, доверенностей, приложений и иных документов.
    2) Если на изображении несколько дат, выбери только дату заключения договора.
    3) Верни ответ строго в одном из форматов:
    - YYYY-MM-DD, если дату можно определить однозначно;
    - ОШИБКА, если дата не найдена или есть сомнения.
    4) Не добавляй пояснения, только итоговое значение.
    """.strip()

IMAGE_MIME_TYPES: dict[str, str] = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".webp": "image/webp",
    ".bmp": "image/bmp",
    ".tif": "image/tiff",
    ".tiff": "image/tiff",
}

RU_MONTH_TO_NUMBER: dict[str, int] = {
    "январь": 1, "января": 1,
    "февраль": 2, "февраля": 2,
    "март": 3, "марта": 3,
    "апрель": 4, "апреля": 4,
    "май": 5, "мая": 5,
    "июнь": 6, "июня": 6,
    "июль": 7, "июля": 7,
    "август": 8, "августа": 8,
    "сентябрь": 9, "сентября": 9,
    "октябрь": 10, "октября": 10,
    "ноябрь": 11, "ноября": 11,
    "декабрь": 12, "декабря": 12,
}

DATE_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\b(?P<year>\d{4})[./-](?P<month>\d{1,2})[./-](?P<day>\d{1,2})\b"),
    re.compile(r"\b(?P<day>\d{1,2})[./-](?P<month>\d{1,2})[./-](?P<year>\d{4})\b"),
    re.compile(
        r"\b(?P<day>\d{1,2})\s+(?P<month>[а-яa-zё]{3,})\s+(?P<year>\d{4})\b",
        re.IGNORECASE,
    ),
)


@dataclass
class ResponseExtractPaidEduContractDateFromImage:
    date: str | None
    success: bool


def extract_paid_edu_contract_date_from_image_with_qwen3_vl(file_name: str) -> ResponseExtractPaidEduContractDateFromImage:
    """Извлекает дату заключения договора из изображения и нормализует её в YYYY-MM-DD."""
    def _guess_image_mime_type(file_name: str) -> str:
        return IMAGE_MIME_TYPES.get(Path(file_name).suffix.lower(), "image/jpeg")

    def _chat_content_to_text(content: object) -> str | None:
        if isinstance(content, str):
            text = content.strip()
            return text or None

        if isinstance(content, list):
            parts: list[str] = []
            for item in content:
                if not isinstance(item, dict):
                    continue
                text_value = item.get("text")
                if isinstance(text_value, str) and text_value.strip():
                    parts.append(text_value.strip())
            joined = "\n".join(parts).strip()
            return joined or None

        return None

    def _parse_month(month_token: str) -> int | None:
        token = month_token.strip().lower().replace("ё", "е").rstrip(".")
        if token.isdigit():
            month = int(token)
            return month if 1 <= month <= 12 else None
        return RU_MONTH_TO_NUMBER.get(token)

    def _normalize_to_iso_date(day: int, month: int, year: int) -> str | None:
        if not (1900 <= year <= 2100):
            return None
        try:
            return datetime(year, month, day).strftime("%Y-%m-%d")
        except ValueError:
            return None

    def _extract_date_candidates(raw_text: str) -> list[str]:
        if not raw_text:
            return []

        prepared_text = re.sub(r"[«»\"“”„‟']", " ", raw_text)
        prepared_text = re.sub(r"\bг\.?\b", " ", prepared_text, flags=re.IGNORECASE)
        prepared_text = re.sub(r"\s+", " ", prepared_text).strip()

        candidates: list[str] = []
        seen: set[str] = set()
        for pattern in DATE_PATTERNS:
            for match in pattern.finditer(prepared_text):
                day = int(match.group("day"))
                year = int(match.group("year"))
                month = _parse_month(match.group("month"))
                if month is None:
                    continue

                normalized = _normalize_to_iso_date(day=day, month=month, year=year)
                if normalized and normalized not in seen:
                    seen.add(normalized)
                    candidates.append(normalized)
        return candidates
    
    def _short_log_text(value: str, limit: int = 220) -> str:
        compact = re.sub(r"\s+", " ", value).strip()
        if len(compact) <= limit:
            return compact
        return f"{compact[:limit]}..."
    


    timeout_sec = float(os.getenv("OCR_VL_TIMEOUT_SEC", "90"))
    max_retries = int(os.getenv("OCR_VL_MAX_RETRIES", "0"))
    model_name = os.getenv("OCR_VL_MODEL", "Qwen/Qwen3-VL-4B-Instruct")

    client = OpenAI(
        base_url=OPENAI_BASE_URL,
        api_key=OPENAI_API_KEY,
        timeout=timeout_sec,
        max_retries=max_retries,
    )

    try:
        with open(file_name, "rb") as f:
            image_base64 = base64.b64encode(f.read()).decode("utf-8")

        image_mime = _guess_image_mime_type(file_name)
        response = client.chat.completions.create(
            model=model_name,
            timeout=timeout_sec,
            temperature=0,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": CONTRACT_DATE_PROMPT},
                    {"type": "image_url", "image_url": {"url": f"data:{image_mime};base64,{image_base64}"}},
                ],
            }],
        )

        raw_text: str | None = None
        if getattr(response, "choices", None):
            raw_text = _chat_content_to_text(response.choices[0].message.content)

        if not raw_text:
            LOGGER.warning("Qwen3-VL returned empty response for contract date: %s", file_name)
            return ResponseExtractPaidEduContractDateFromImage(date=None, success=False)

        if raw_text.strip().upper() == "ОШИБКА":
            return ResponseExtractPaidEduContractDateFromImage(date=None, success=False)

        date_candidates = _extract_date_candidates(raw_text)
        if len(date_candidates) == 1:
            return ResponseExtractPaidEduContractDateFromImage(date=date_candidates[0], success=True)

        if len(date_candidates) > 1:
            LOGGER.warning(
                "Qwen3-VL returned multiple date candidates for %s: %s; response=%s",
                file_name,
                date_candidates,
                _short_log_text(raw_text),
            )
            return ResponseExtractPaidEduContractDateFromImage(date=None, success=False)

        LOGGER.warning(
            "Qwen3-VL did not return a parsable contract date for %s: %s",
            file_name,
            _short_log_text(raw_text),
        )
        return ResponseExtractPaidEduContractDateFromImage(date=None, success=False)

    except Exception as e:
        LOGGER.error(
            "Qwen3-VL contract-date OCR failed for %s (model=%s, timeout=%ss, retries=%s): %s",
            file_name,
            model_name,
            timeout_sec,
            max_retries,
            e,
        )
        return ResponseExtractPaidEduContractDateFromImage(date=None, success=False)
        

def _parse_bool_answer(raw_response: object) -> bool | None:
    if isinstance(raw_response, list):
        if not raw_response:
            return None
        raw_response = raw_response[0]

    if isinstance(raw_response, dict):
        raw_text_value = raw_response.get("text") or raw_response.get("content")
        if raw_text_value is None:
            return None
        raw_text = str(raw_text_value)
    else:
        raw_text = str(raw_response)

    answer_upper = raw_text.strip().upper()
    compact_answer = re.sub(r"[^A-ZА-Я]+", " ", answer_upper).strip()
    tokens = compact_answer.split()

    for token in tokens:
        if token in {"TRUE", "ДА", "YES"}:
            return True
        if token in {"FALSE", "НЕТ", "NO"}:
            return False

    if "TRUE" in answer_upper:
        return True
    if "FALSE" in answer_upper:
        return False
    return None


def _is_empty_or_truncated_response(raw_response: object) -> bool:
    if not isinstance(raw_response, list) or not raw_response:
        return False

    first_item = raw_response[0]
    if isinstance(first_item, dict):
        text_value = str(first_item.get("text") or first_item.get("content") or "").strip()
        finish_reason = str(
            first_item.get("finish_reason")
            or first_item.get("finishReason")
            or ""
        ).strip().lower()
        reasoning_content = str(first_item.get("reasoning_content") or "").strip()

        if finish_reason in {"length", "max_tokens"}:
            return True
        if not text_value and reasoning_content:
            return True
        if not text_value:
            return True
    return False


def check_university_name_llm(cell_university_name: str, university_name: str, lm: dspy.LM | None = None) -> bool:
    """Отправляет запрос LLM для сравнения названий университетов."""
    
    # Быстрая проверка на пустые значения
    if not cell_university_name or not university_name:
        return False
    
    # Проверка на токены ошибки
    if university_name.strip().upper() == "ОШИБКА":
        return False
    
    local_api_base = normalize_local_api_base(os.environ["API_URL"])
    if lm is None:
        lm = build_local_text_lm()
    else:
        lm_model = str(getattr(lm, "model", "")).strip()
        lm_api_base = str(
            getattr(lm, "kwargs", {}).get("api_base")
            or getattr(lm, "kwargs", {}).get("base_url")
            or ""
        ).strip()
        if lm_model != LOCAL_TEXT_LLM_MODEL or lm_api_base.rstrip("/") != local_api_base.rstrip("/"):
            LOGGER.info("Переданный LLM не локальный, используем локальный LLM-сервер для сравнения ВУЗов.")
            lm = build_local_text_lm()
    
    
    # Формируем промпт прямо здесь
    prompt = f"""
        Ты помогаешь сверять названия университетов.

        Задача: определи, относятся ли два названия к одному и тому же ВУЗу.

        Правила:
        1) Игнорируй различия в регистре, пробелах, пунктуации.
        2) Сокращения и полные названия одного ВУЗа считай совпадением (например: "МГУ" и "Московский государственный университет").
        3) Если в названиях совпадает ключевое имя (например в кавычках),
           считай что это один ВУЗ, даже если различаются служебные слова
           вроде "федеральный/национальный/государственный/исследовательский".
        4) Учитывай возможные OCR-опечатки и ошибки извлечения текста:
           - замена/пропуск/добавление 1-2 букв;
           - близкие варианты написания имён и слов.
           Если почти всё название совпадает, а различие только в вероятной опечатке,
           считай что это один ВУЗ.
           Пример: "Патрика Лумумбы" и "Патриса Лумумбы" — это один и тот же ВУЗ.
        5) Если названия явно разные по ключевой части — верни False.
        6) Верни ТОЛЬКО True или False, без объяснений.

        Название из Excel: {cell_university_name}
        Название из документа: {university_name}

        Ответ (True/False):
        """.strip()
    
    max_attempts = max(1, int(os.getenv("UNIVERSITY_CHECK_LLM_RETRIES", "3")))
    retry_delay_sec = max(0.0, float(os.getenv("UNIVERSITY_CHECK_LLM_RETRY_DELAY_SEC", "0.8")))
    try:
        for attempt in range(1, max_attempts + 1):
            lm_for_attempt = lm if attempt == 1 else build_local_text_lm()
            attempt_model = str(getattr(lm_for_attempt, "model", LOCAL_TEXT_LLM_MODEL))

            LOGGER.info(
                "Отправили запрос с промптом на локальный LLM-сервер (попытка %d/%d, модель=%s).",
                attempt,
                max_attempts,
                attempt_model,
            )
            response = lm_for_attempt(prompt, temperature=0)
            LOGGER.info("Получили ответ от локального LLM-сервера.")
            parsed_answer = _parse_bool_answer(response)

            LOGGER.info("Парсер ответ.")
            if parsed_answer is not None:
                return parsed_answer

            should_retry = (
                attempt < max_attempts
                and _is_empty_or_truncated_response(response)
            )
            if should_retry:
                LOGGER.warning(
                    "LLM вернул пустой/обрезанный ответ (попытка %d/%d): %r. Повторяем запрос.",
                    attempt,
                    max_attempts,
                    response,
                )
                if retry_delay_sec > 0:
                    sleep(retry_delay_sec)
                continue

            LOGGER.warning("LLM вернул нераспознанный ответ при сравнении ВУЗов: %r", response)
            return False

        return False
            
    except Exception as e:
        LOGGER.warning(
            f"Ошибка при сравнении названий ВУЗов через локальный LLM-сервер: {e}"
        )
        return False




    


if __name__ == "__main__":
    start = perf_counter()
    resp1 = extract_text_from_image_with_qwen3_vl("test_images/01.02.2025 130291.jpg")
    t1 = perf_counter() - start
    print(resp1.text)

    start = perf_counter()
    resp2 = extract_text_from_image("test_images/01.02.2025 130291.jpg")
    t2 = perf_counter() - start
    print(resp2.text)

    print(f"extract_text_from_image_with_qwen3_vl: {t1:.3f}s")
    print(f"extract_text_from_image: {t2:.3f}s")
