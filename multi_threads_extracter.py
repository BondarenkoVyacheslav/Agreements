from __future__ import annotations

from pathlib import Path
import json
import logging
import os
import re
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

import dspy

import transfer_excel_data as transfer_excel_module
from extract_field_data_with_qwen_vl import extract_filed_data_with_qwen_vl
from extract_fields import (
    ExtractedFields,
    Extractor,
    build_trainset,
    compile_extractor,
    extract_fields_from_image_with_qwen3_vl_ocr,
    extract_fields_from_text_content,
)
from ocr_client import (
    build_local_text_lm,
    extract_paid_edu_contract_date_from_image_with_qwen3_vl,
    extract_text_from_image_with_qwen3_vl,
)
from prompt_settings import ERROR_TOKEN

DIRECTORY = Path("А2/МинОбр")
OUTPUT_WORKBOOK_PATH = Path("1448 декабрь в Сбер на проверку Выгрузка.xlsx")
TRAIN_JSONL: Path | None = None  # например: Path("train_data.jsonl")
MAX_FILES_TO_PROCESS: int | None = None
MAX_WORKERS = 5
LOGGER = logging.getLogger(__name__)
PDF_EXTENSION = ".pdf"
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff")
SUPPORTED_DOCUMENT_EXTENSIONS = IMAGE_EXTENSIONS + (PDF_EXTENSION,)
NUMBER_FIRST_STEM_PATTERN = re.compile(r"^(?P<number>.+?)[_ ]+(?P<date>\d{1,2}\.\d{1,2}\.\d{4})$")
DATE_FIRST_STEM_PATTERN = re.compile(r"^(?P<date>\d{1,2}\.\d{1,2}\.\d{4})[_ ]+(?P<number>.+?)$")
PDF_RENDERER_PATH = Path("/usr/bin/pdftoppm")
PDF_RENDER_DPI = int(os.getenv("PDF_RENDER_DPI", "200"))

# Блокировка для потокобезопасной работы с Excel
excel_lock = threading.Lock()
thread_state = threading.local()


def _build_lm() -> dspy.LM:
    return build_local_text_lm()


def get_thread_lm() -> dspy.LM:
    lm = getattr(thread_state, "lm", None)
    if lm is None:
        lm = _build_lm()
        thread_state.lm = lm
    return lm


def configure_logging() -> None:
    log_level_name = os.getenv("LOG_LEVEL", "INFO").upper()
    log_level = getattr(logging, log_level_name, logging.INFO)
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


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


def _configure_output_workbook_path() -> None:
    transfer_excel_module.OUTPUT_PATH = OUTPUT_WORKBOOK_PATH


def _list_input_documents(directory: Path) -> list[Path]:
    documents: list[Path] = []
    for path in sorted(directory.iterdir()):
        if not path.is_file():
            continue
        if path.suffix.lower() not in SUPPORTED_DOCUMENT_EXTENSIONS:
            LOGGER.warning("Пропуск файла с неподдерживаемым расширением: %s", path.name)
            continue
        documents.append(path)
    return documents


def _parse_agreement_identity(document_path: Path) -> str | None:
    return transfer_excel_module.extract_credit_contract_number_from_document_name(document_path)


def _document_has_embedded_date(document_path: Path) -> bool:
    stem = document_path.stem.rstrip(".").strip()
    return bool(
        NUMBER_FIRST_STEM_PATTERN.fullmatch(stem)
        or DATE_FIRST_STEM_PATTERN.fullmatch(stem)
    )


def _document_priority(document_path: Path) -> tuple[int, int, str]:
    return (
        0 if _document_has_embedded_date(document_path) else 1,
        len(document_path.name),
        document_path.name,
    )


def _deduplicate_documents_by_contract_number(documents: list[Path]) -> list[Path]:
    grouped_documents: dict[str, list[Path]] = {}
    for document_path in documents:
        agreement_number = _parse_agreement_identity(document_path)
        if agreement_number is None:
            LOGGER.warning("Пропуск файла с невалидным именем: %s", document_path.name)
            continue
        grouped_documents.setdefault(agreement_number, []).append(document_path)

    selected_documents: list[Path] = []
    for agreement_number, contract_documents in grouped_documents.items():
        canonical_document = min(contract_documents, key=_document_priority)
        selected_documents.append(canonical_document)

        if len(contract_documents) > 1:
            skipped_documents = sorted(
                path.name for path in contract_documents
                if path != canonical_document
            )
            LOGGER.warning(
                "Найдено %d файлов для договора %s. Выбран %s, пропущены: %s",
                len(contract_documents),
                agreement_number,
                canonical_document.name,
                ", ".join(skipped_documents),
            )

    return sorted(selected_documents, key=lambda path: path.name)


def _render_pdf_pages(pdf_path: Path, output_dir: Path) -> list[Path]:
    if not PDF_RENDERER_PATH.exists():
        raise FileNotFoundError(f"PDF renderer not found: {PDF_RENDERER_PATH}")

    output_prefix = output_dir / "page"
    completed = subprocess.run(
        [str(PDF_RENDERER_PATH), "-png", "-r", str(PDF_RENDER_DPI), str(pdf_path), str(output_prefix)],
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        stderr = ((completed.stderr or "").strip() or (completed.stdout or "").strip() or "unknown pdftoppm error")
        raise RuntimeError(
            f"pdftoppm failed for {pdf_path.name} with code {completed.returncode}: {stderr}"
        )

    rendered_pages = sorted(output_dir.glob("page-*.png"))
    if not rendered_pages:
        raise FileNotFoundError(f"PDF pages were not rendered for {pdf_path}")
    return rendered_pages


def _ocr_rendered_pages(rendered_pages: list[Path], thread_name: str) -> str:
    page_texts: list[str] = []

    for page_index, rendered_page in enumerate(rendered_pages, start=1):
        LOGGER.info(
            "[%s] OCR PDF-страницы %d/%d: %s",
            thread_name,
            page_index,
            len(rendered_pages),
            rendered_page.name,
        )
        response = extract_text_from_image_with_qwen3_vl(str(rendered_page))
        if not response.success or not response.text:
            LOGGER.warning(
                "[%s] OCR не вернул текст для PDF-страницы %d (%s).",
                thread_name,
                page_index,
                rendered_page.name,
            )
            continue

        page_texts.append(response.text.strip())

    return "\n\n".join(text for text in page_texts if text).strip()


def _should_retry_contract_date(extracted_fields: ExtractedFields) -> bool:
    if extracted_fields.paid_edu_contract_date != ERROR_TOKEN:
        return False

    return any(
        getattr(extracted_fields, field_name) != ERROR_TOKEN
        for field_name in (
            "university_name",
            "student_fio",
            "customer_fio",
            "paid_edu_contract_number",
            "specialty_code",
        )
    )


def _fill_pdf_contract_date_from_pages(
    extracted_fields: ExtractedFields,
    rendered_pages: list[Path],
    thread_name: str,
) -> ExtractedFields:
    if not _should_retry_contract_date(extracted_fields):
        return extracted_fields

    for page_index, rendered_page in enumerate(rendered_pages, start=1):
        LOGGER.info(
            "[%s] Повторная попытка извлечения даты из PDF-страницы %d/%d: %s",
            thread_name,
            page_index,
            len(rendered_pages),
            rendered_page.name,
        )
        response = extract_paid_edu_contract_date_from_image_with_qwen3_vl(str(rendered_page))
        if response.success and response.date and response.date != ERROR_TOKEN:
            extracted_fields.paid_edu_contract_date = response.date
            LOGGER.info(
                "[%s] Дата договора найдена на PDF-странице %d: %s",
                thread_name,
                page_index,
                response.date,
            )
            break

    return extracted_fields


def process_single_file(args: tuple) -> bool:
    """
    Обработка одного документа в отдельном потоке.
    :param args: кортеж (index, document_path, compiled_extractor)
    :return: True если успешно, False если ошибка
    """
    i, document_path, compiled_extractor = args
    thread_name = threading.current_thread().name
    
    try:
        thread_lm = get_thread_lm()
        LOGGER.info("[%s] Документ №%d: %s", thread_name, i, document_path.name)

        educational_loan_agreement_number = _parse_agreement_identity(document_path)
        if educational_loan_agreement_number is None:
            LOGGER.warning("[%s] Пропуск файла с невалидным именем: %s", thread_name, document_path.name)
            return False

        if document_path.suffix.lower() == PDF_EXTENSION:
            with tempfile.TemporaryDirectory(prefix="agreements_pdf_", dir="/tmp") as temp_dir_name:
                rendered_pages = _render_pdf_pages(document_path, Path(temp_dir_name))
                LOGGER.info(
                    "[%s] PDF %s отрендерен в %d страниц(ы).",
                    thread_name,
                    document_path.name,
                    len(rendered_pages),
                )
                merged_text = _ocr_rendered_pages(rendered_pages, thread_name)
                with dspy.context(lm=thread_lm):
                    result = extract_fields_from_text_content(compiled_extractor, merged_text)
                result = _fill_pdf_contract_date_from_pages(result, rendered_pages, thread_name)
        else:
            with dspy.context(lm=thread_lm):
                result = extract_fields_from_image_with_qwen3_vl_ocr(compiled_extractor, str(document_path))
            result = extract_filed_data_with_qwen_vl(document_path.stem, result, document_path)

        LOGGER.info(
            f"[{thread_name}] Поля извлечены для договора {educational_loan_agreement_number}"
        )
        LOGGER.debug(f"[{thread_name}] Извлечённые данные: {result}")

        # Блокировка для потокобезопасной работы с Excel
        with excel_lock:
            _configure_output_workbook_path()

            # Далее заполняем извлеченные поля в соответствующие колонки
            if not transfer_excel_module.transfer_extracted_data_and_logic_to_excel(
                educational_loan_agreement_number, 
                result,
                thread_lm
            ):
                LOGGER.error(
                    f"[{thread_name}] Проблемы с заполнением полей от LLM/логики вывода для договора "
                    f"{educational_loan_agreement_number}."
                )
            LOGGER.info(
                f"[{thread_name}] Заполнены извлечённые поля и сформирован вывод ИИ для договора "
                f"{educational_loan_agreement_number}."
            )
        
        return True
        
    except Exception:
        LOGGER.exception(
            f"[{thread_name}] Необработанная ошибка при обработке файла {document_path}. Переходим к следующему."
        )
        return False


def main() -> int:
    """
    Тут собираем все модули и реализуем всю логику MVP с мультипоточностью.
    :return: int
    :rtype: int
    """
    configure_logging()
    _configure_output_workbook_path()

    # Заполняем выгрузку данными которые нам передали
    transfer_excel_module.transfer_excel_data()

    # Извлекаем все необходимые поля
    # Настраиваем LLM для DSPy (один раз)
    startup_lm = _build_lm()
    dspy.configure(lm=startup_lm)

    compiled_extractor = build_compiled_extractor(TRAIN_JSONL)

    if not DIRECTORY.exists():
        raise FileNotFoundError(f"Documents directory not found: {DIRECTORY}")

    all_documents = _list_input_documents(DIRECTORY)
    if not all_documents:
        raise FileNotFoundError(f"No supported documents found in: {DIRECTORY}")

    unique_documents = _deduplicate_documents_by_contract_number(all_documents)
    if not unique_documents:
        raise FileNotFoundError(f"No documents with valid contract names found in: {DIRECTORY}")

    if MAX_FILES_TO_PROCESS is not None:
        unique_documents = unique_documents[:MAX_FILES_TO_PROCESS]

    LOGGER.info("Найдено %d поддерживаемых документов в %s.", len(all_documents), DIRECTORY)
    LOGGER.info("После дедупликации будет обработано %d файлов в %d потоках.", len(unique_documents), MAX_WORKERS)

    # Подготовка аргументов для потоков
    tasks = [(i, document_path, compiled_extractor) for i, document_path in enumerate(unique_documents, start=1)]
    
    # Запуск пула потоков
    success_count = 0
    error_count = 0
    
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # Отправляем все задачи в пул
        future_to_task = {executor.submit(process_single_file, task): task for task in tasks}
        
        # Обрабатываем результаты по мере завершения
        for future in as_completed(future_to_task):
            task = future_to_task[future]
            i, document_path, _ = task
            try:
                result = future.result()
                if result:
                    success_count += 1
                    LOGGER.info(f"Файл №{i} ({document_path.name}) успешно обработан.")
                else:
                    error_count += 1
                    LOGGER.warning(f"Файл №{i} ({document_path.name}) обработан с ошибками.")
            except Exception as e:
                error_count += 1
                LOGGER.exception(f"Файл №{i} ({document_path.name}) вызвал исключение: {e}")

    LOGGER.info("=" * 50)
    LOGGER.info("Обработка завершена!")
    LOGGER.info("Успешно: %d, Ошибок: %d, Всего: %d", success_count, error_count, len(unique_documents))
    LOGGER.info("=" * 50)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
