import base64
import logging
import os
from time import perf_counter
from dataclasses import dataclass

import requests
from openai import OpenAI
import dspy


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
    

import dspy
import os
from typing import Optional


def check_university_name_llm(cell_university_name: str, university_name: str) -> bool:
    """
    Отправляет запрос LLM для сравнения названий университетов.
    
    :param cell_university_name: Название университета из Excel-ячейки (эталон)
    :param university_name: Название университета, извлечённое LLM из документа
    :return: True если названия относятся к одному ВУЗу, иначе False
    """
    
    # Быстрая проверка на пустые значения
    if not cell_university_name or not university_name:
        return False
    
    # Проверка на токены ошибки
    if university_name.strip().upper() == "ОШИБКА":
        return False
    
    # Инициализация LLM
    lm = dspy.LM(
        "openrouter/qwen/qwen3-30b-a3b", 
        api_key=os.environ["OPEN_ROUTER_API_KEY"]
    )
    dspy.configure(lm=lm)
    
    # Формируем промпт прямо здесь
    prompt = f"""
        Ты помогаешь сверять названия университетов.

        Задача: определи, относятся ли два названия к одному и тому же ВУЗу.

        Правила:
        1) Игнорируй различия в регистре, пробелах, пунктуации.
        2) Сокращения и полные названия одного ВУЗа считай совпадением (например: "МГУ" и "Московский государственный университет").
        3) Если названия явно разные — верни False.
        4) Верни ТОЛЬКО True или False, без объяснений.

        Название из Excel: {cell_university_name}
        Название из документа: {university_name}

        Ответ (True/False):
        """.strip()
    
    try:
        # Отправляем запрос к LLM
        response = lm(prompt, max_tokens=10)
        answer = str(response[0]).strip().upper()
        
        # Парсим ответ
        if answer == "TRUE":
            return True
        elif answer == "FALSE":
            return False
        else:
            # Если ответ нечёткий — консервативно возвращаем False
            return False
            
    except Exception as e:
        import logging
        logging.getLogger(__name__).warning(
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
