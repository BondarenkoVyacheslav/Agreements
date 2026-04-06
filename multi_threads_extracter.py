from pathlib import Path
import os
import json
import logging
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import dspy
from transfer_excel_data import transfer_excel_data, transfer_reporting_data_to_excel, transfer_extracted_data_and_logic_to_excel
from extract_fields import (
    build_trainset,
    compile_extractor,
    extract_fields_from_image_with_qwen3_vl_ocr,
    ExtractedFields,
    Extractor,
)
from extract_field_data_with_qwen_vl import extract_filed_data_with_qwen_vl

DIRECTORY = Path("agreements")
TRAIN_JSONL: Path | None = None  # например: Path("train_data.jsonl")
MAX_FILES_TO_PROCESS = 100
MAX_WORKERS = 5  # Количество потоков
LOGGER = logging.getLogger(__name__)
OPENROUTER_MODEL = "openrouter/qwen/qwen3-30b-a3b"
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff")

# Блокировка для потокобезопасной работы с Excel
excel_lock = threading.Lock()
thread_state = threading.local()

REPORTING_DATES_BY_MONTH = {
    "01": "ЯНВАРЬ!",
    "02": "ФЕВРАЛЬ!",
    "03": "МАРТ!",
    "04": "АПРЕЛЬ!",
    "05": "МАЙ!",
    "06": "ИЮНЬ!",
    "07": "ИЮЛЬ;АВГУСТ;СЕНТЯБРЬ",
    "08": "АВГУСТ;СЕНТЯБРЬ",
    "09": "СЕНТЯБРЬ",
    "10": "ОКТЯБРЬ!",
    "11": "НОЯБРЬ!",
    "12": "ДЕКАБРЬ!",
}


def _build_lm() -> dspy.LM:
    return dspy.LM(OPENROUTER_MODEL, api_key=os.environ["OPEN_ROUTER_API_KEY"])


def get_thread_lm() -> dspy.LM:
    lm = getattr(thread_state, "lm", None)
    if lm is None:
        lm = _build_lm()
        thread_state.lm = lm
    return lm


def parse_contract_month(contract_date: str | None) -> str | None:
    if not contract_date:
        return None
    value = contract_date.strip()
    if not value or value.upper() == "ОШИБКА":
        return None

    formats = (
        "%Y-%m-%d",
        "%d.%m.%Y",
        "%d-%m-%Y",
        "%Y.%m.%d",
        "%d/%m/%Y",
        "%Y/%m/%d",
    )

    for date_value in (value, value.split()[0]):
        for date_format in formats:
            try:
                return datetime.strptime(date_value, date_format).strftime("%m")
            except ValueError:
                continue

    return None


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


def _list_input_images(directory: Path) -> list[Path]:
    return sorted(
        path for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def process_single_file(args: tuple) -> bool:
    """
    Обработка одного изображения в отдельном потоке.
    :param args: кортеж (index, image_file_path, compiled_extractor)
    :return: True если успешно, False если ошибка
    """
    i, image_path, compiled_extractor = args
    thread_name = threading.current_thread().name
    
    try:
        thread_lm = get_thread_lm()
        LOGGER.info(f"[{thread_name}] Изображение №{i}: {image_path.name}")
        
        stem = image_path.stem
        stem_parts = stem.replace("_", "  ").split(maxsplit=1)
        if len(stem_parts) < 2:
            LOGGER.error(f"[{thread_name}] Пропуск файла с невалидным именем: {image_path.name}")
            return False
        
        educational_loan_agreement_date, educational_loan_agreement_number = stem_parts

        with dspy.context(lm=thread_lm):
            result: ExtractedFields = extract_fields_from_image_with_qwen3_vl_ocr(compiled_extractor, str(image_path))
        LOGGER.info(
            f"[{thread_name}] Поля извлечены для договора {educational_loan_agreement_number} от {educational_loan_agreement_date}"
        )
        LOGGER.debug(f"[{thread_name}] Извлечённые данные: {result}")

        # Тут необходимо доп. проверка даты, если дата не была поймана с первого раза ocr-ом, необходимо еще раз попробовать qwen_vl
        result = extract_filed_data_with_qwen_vl(stem, result, image_path)


        # Заполняем поле 9 согласно дате заключения договора
        month = parse_contract_month(result.paid_edu_contract_date)
        reporting_dates = REPORTING_DATES_BY_MONTH.get(month)
        if month is None:
            LOGGER.warning(
                f"[{thread_name}] Не удалось определить месяц по paid_edu_contract_date={result.paid_edu_contract_date}"
            )
        elif reporting_dates is None:
            LOGGER.warning(
                f"[{thread_name}] Месяц {month} вне ожидаемого диапазона (07-09), paid_edu_contract_date={result.paid_edu_contract_date}"
            )

        # Блокировка для потокобезопасной работы с Excel
        with excel_lock:
            if not transfer_reporting_data_to_excel(
                educational_loan_agreement_number, 
                educational_loan_agreement_date, 
                reporting_dates
            ):
                LOGGER.error(
                    f"[{thread_name}] Договор с номером образовательного кредита {educational_loan_agreement_number} "
                    f"и датой {educational_loan_agreement_date} не найден при заполнении reporting-данных."
                )
            LOGGER.info(f"[{thread_name}] Определены reporting_dates: {reporting_dates}")

            # Далее заполняем извлеченные поля в соответствующие колонки
            if not transfer_extracted_data_and_logic_to_excel(
                educational_loan_agreement_number, 
                educational_loan_agreement_date, 
                result,
                thread_lm
            ):
                LOGGER.error(
                    f"[{thread_name}] Проблемы с заполнением полей от LLM/логики вывода для договора "
                    f"{educational_loan_agreement_number} от {educational_loan_agreement_date}."
                )
            LOGGER.info(
                f"[{thread_name}] Заполнены извлечённые поля и сформирован вывод ИИ для договора "
                f"{educational_loan_agreement_number} от {educational_loan_agreement_date}."
            )
        
        return True
        
    except Exception:
        LOGGER.exception(
            f"[{thread_name}] Необработанная ошибка при обработке файла {image_path}. Переходим к следующему."
        )
        return False


def main() -> int:
    """
    Тут собираем все модули и реализуем всю логику MVP с мультипоточностью.
    :return: int
    :rtype: int
    """
    configure_logging()

    # Заполняем выгрузку данными которые нам передали
    transfer_excel_data()

    # Извлекаем все необходимые поля
    # Настраиваем LLM для DSPy (один раз)
    startup_lm = _build_lm()
    dspy.configure(lm=startup_lm)

    compiled_extractor = build_compiled_extractor(TRAIN_JSONL)

    if not DIRECTORY.exists():
        raise FileNotFoundError(f"Images directory not found: {DIRECTORY}")

    all_images = _list_input_images(DIRECTORY)
    if not all_images:
        raise FileNotFoundError(f"No image files found in: {DIRECTORY}")
    
    # all_images = all_images[:MAX_FILES_TO_PROCESS]
    
    LOGGER.info("Найдено %d изображений.", len(all_images))
    # all_images = all_images[:MAX_FILES_TO_PROCESS]
    LOGGER.info("Будет обработано %d файлов в %d потоках.", len(all_images), MAX_WORKERS)

    # Подготовка аргументов для потоков
    tasks = [(i, image_path, compiled_extractor) for i, image_path in enumerate(all_images)]
    
    # Запуск пула потоков
    success_count = 0
    error_count = 0
    
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # Отправляем все задачи в пул
        future_to_task = {executor.submit(process_single_file, task): task for task in tasks}
        
        # Обрабатываем результаты по мере завершения
        for future in as_completed(future_to_task):
            task = future_to_task[future]
            i, image_path, _ = task
            try:
                result = future.result()
                if result:
                    success_count += 1
                    LOGGER.info(f"Файл №{i} ({image_path.name}) успешно обработан.")
                else:
                    error_count += 1
                    LOGGER.warning(f"Файл №{i} ({image_path.name}) обработан с ошибками.")
            except Exception as e:
                error_count += 1
                LOGGER.exception(f"Файл №{i} ({image_path.name}) вызвал исключение: {e}")

    LOGGER.info("=" * 50)
    LOGGER.info("Обработка завершена!")
    LOGGER.info("Успешно: %d, Ошибок: %d, Всего: %d", success_count, error_count, len(all_images))
    LOGGER.info("=" * 50)
    LOGGER.info("slavik........")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
