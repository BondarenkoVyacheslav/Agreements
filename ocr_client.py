import requests
from dataclasses import dataclass


BASE_URL = "https://ocr.g-309.ru" 

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


print(extract_text_from_image("test_files/test3_p.jpg").text)