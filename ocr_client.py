import base64
import logging
import os
import re
from time import perf_counter, sleep
from dataclasses import dataclass

import requests
from openai import OpenAI
import dspy

LOGGER = logging.getLogger(__name__)

BASE_URL = "https://ocr.g-309.ru"
OPENAI_BASE_URL = "http://10.14.49.32:31751/v1"
OPENAI_API_KEY = ""

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
    

def _parse_bool_answer(raw_response: object) -> bool | None:
    if isinstance(raw_response, list):
        if not raw_response:
            return None
        raw_response = raw_response[0]

    if isinstance(raw_response, dict):
        raw_text = str(
            raw_response.get("text")
            or raw_response.get("content")
            or raw_response
        )
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


def _normalize_openrouter_model(model_name: str) -> str:
    value = str(model_name).strip()
    if not value:
        return value
    if value.startswith("openrouter/"):
        return value
    return f"openrouter/{value}"


def check_university_name_llm(cell_university_name: str, university_name: str, lm: dspy.LM | None = None) -> bool:
    """Отправляет запрос LLM для сравнения названий университетов."""
    
    # Быстрая проверка на пустые значения
    if not cell_university_name or not university_name:
        return False
    
    # Проверка на токены ошибки
    if university_name.strip().upper() == "ОШИБКА":
        return False
    
    # Инициализация LLM
    model_chain = [
        _normalize_openrouter_model(os.getenv("UNIVERSITY_CHECK_LLM_MODEL_1", "openrouter/qwen/qwen3-30b-a3b")),
        _normalize_openrouter_model(os.getenv("UNIVERSITY_CHECK_LLM_MODEL_2", "openai/gpt-oss-120b")),
        _normalize_openrouter_model(os.getenv("UNIVERSITY_CHECK_LLM_MODEL_3", "google/gemini-3-flash-preview")),
    ]
    model_chain = [model for model in model_chain if model]

    if lm is None:
        lm = dspy.LM(
            model_chain[0],
            api_key=os.environ["OPEN_ROUTER_API_KEY"]
        )
    
    
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
    fallback_lms_by_model: dict[str, dspy.LM] = {}

    try:
        for attempt in range(1, max_attempts + 1):
            if attempt == 1:
                lm_for_attempt = lm
                attempt_model = str(getattr(lm_for_attempt, "model", model_chain[0]))
            else:
                model_index = min(attempt - 1, len(model_chain) - 1)
                attempt_model = model_chain[model_index]
                if attempt_model not in fallback_lms_by_model:
                    fallback_lms_by_model[attempt_model] = dspy.LM(
                        attempt_model,
                        api_key=os.environ["OPEN_ROUTER_API_KEY"],
                    )
                lm_for_attempt = fallback_lms_by_model[attempt_model]

            LOGGER.info(
                "Отправили запрос с промптом на OpenRouter (попытка %d/%d, модель=%s).",
                attempt,
                max_attempts,
                attempt_model,
            )
            response = lm_for_attempt(prompt, temperature=0)
            LOGGER.info("Получили ответ от OpenRouter.")
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
            f"Ошибка при сравнении названий ВУЗов через LLM: {e}"
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
