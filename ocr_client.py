import requests
from dataclasses import dataclass
from openai import OpenAI
import base64
from time import perf_counter


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
    client = OpenAI(base_url=OPENAI_BASE_URL, api_key=OPENAI_API_KEY)
    with open(file_name, "rb") as f:
        image_base64 = base64.b64encode(f.read()).decode("utf-8")

    response = client.chat.completions.create(
        model="Qwen/Qwen3-VL-4B-Instruct",
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": "Extract all text from this image. Return only the text."},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}},
            ],
        }],
    )
    text = response.choices[0].message.content
    return OcrClientResponse(text=text, success=bool(text))


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