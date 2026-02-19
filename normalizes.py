from typing import Any
from prompt_settings import ERROR_TOKEN
import re

DATE_PATTERN = re.compile(r"\d{4}-\d{2}-\d{2}")
SPECIALTY_PATTERN = re.compile(r"\d{2}\.\d{2}\.\d{2}")

def _as_clean_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()

def _normalize_value(value: Any) -> str:
    text = _as_clean_text(value)
    if not text or text == "{}":
        return ERROR_TOKEN
    return text

def _normalize_date(value: Any) -> str:
    text = _normalize_value(value)
    if text == ERROR_TOKEN:
        return ERROR_TOKEN
    if DATE_PATTERN.fullmatch(text):
        return text
    return ERROR_TOKEN


def _normalize_specialty(value: Any) -> str:
    text = _normalize_value(value)
    if text == ERROR_TOKEN:
        return ERROR_TOKEN
    if SPECIALTY_PATTERN.fullmatch(text):
        return text
    return ERROR_TOKEN


def _normalize_evidence(value: Any, extracted_value: str) -> str:
    if extracted_value == ERROR_TOKEN:
        return ERROR_TOKEN
    text = _as_clean_text(value)
    if not text or text == "{}":
        return ERROR_TOKEN
    return text