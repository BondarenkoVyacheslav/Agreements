from __future__ import annotations

from contextlib import closing
from pathlib import Path

from openpyxl import load_workbook
from openpyxl.styles import Alignment, Font

from extract_fields import ExtractedFields
from utils import _norm as _norm_text, _norm_code

SOURCE_PATH = Path("СВОД СБЕР июль-сентябрь.xlsx")
TEMPLATE_PATH = Path("Шаблон_выгрузки.xlsx")
OUTPUT_PATH = Path("СВОД_СБЕР_выгрузка.xlsx")
DATA_START_ROW = 2

def transfer_excel_data() -> int:
    if not SOURCE_PATH.exists():
        print(f"File not found: {SOURCE_PATH}")
        return 1
    if not TEMPLATE_PATH.exists():
        print(f"File not found: {TEMPLATE_PATH}")
        return 1

    with closing(load_workbook(SOURCE_PATH, read_only=True, data_only=True)) as wb_src:
        ws_src = wb_src.active
        wb_out = load_workbook(TEMPLATE_PATH)
        ws_out = wb_out.active

        alignment = Alignment(horizontal="center", vertical="center")
        font = Font(name="Times New Roman", size=12)

        out_row = DATA_START_ROW
        for i, row in enumerate(ws_src.iter_rows(values_only=True)):
            if i == 0 or i == 1:
                continue

            position = row[0]
            university_name = row[1]
            full_name = row[2]
            educational_loan_agreement_with_date_and_number = row[3]
            code_of_study = row[6]
            unique_id_agreement = row[7]
            adjustment_sign = row[8]

            educational_loan_agreement_date = educational_loan_agreement_with_date_and_number.split()[0]
            educational_loan_agreement_number = educational_loan_agreement_with_date_and_number.split()[1]

            values = [
                position,
                university_name,
                full_name,
                educational_loan_agreement_number,
                educational_loan_agreement_date,
                code_of_study,
                unique_id_agreement,
                adjustment_sign,
            ]

            for col_idx, value in enumerate(values, start=1):
                cell = ws_out.cell(row=out_row, column=col_idx, value=value)
                cell.alignment = alignment
                cell.font = font

            out_row += 1

        wb_out.save(OUTPUT_PATH)
        wb_out.close()

    return 0


def transfer_reporting_data_to_excel(edu_loan_agr_num: str, edu_loan_agr_date: str, reporting_dates: str = "НЕ ОПРЕДЕЛЕНО") -> bool:
    """Заполняем для нужного договора 9 графу с месяцами"""
    if not OUTPUT_PATH.exists():
        if transfer_excel_data() != 0:
            print(f"File not found: {OUTPUT_PATH}")
            return False

    from datetime import date, datetime

    def _norm(value: object) -> str:
        if value is None:
            return ""
        if isinstance(value, (datetime, date)):
            return value.strftime("%Y-%m-%d")
        return str(value).strip()

    def _safe_value(value: object) -> str:
        if value is None:
            return "ОШИБКА"
        if isinstance(value, (datetime, date)):
            return value.strftime("%Y-%m-%d")
        text = str(value).strip()
        return text if text else "ОШИБКА"

    target_date = _norm(edu_loan_agr_date)
    target_num = _norm(edu_loan_agr_num)

    try:
        wb_out = load_workbook(OUTPUT_PATH)
    except TypeError as exc:
        if "extLst" in str(exc):
            if transfer_excel_data() != 0:
                print(f"Failed to rebuild output: {OUTPUT_PATH}")
                return False
            wb_out = load_workbook(OUTPUT_PATH)
        else:
            raise
    ws_out = wb_out.active

    alignment = Alignment(horizontal="center", vertical="center")
    font = Font(name="Times New Roman", size=12)

    found = False
    for row in ws_out.iter_rows(min_row=DATA_START_ROW):
        cell_num = row[3]  # 4-й столбец
        cell_date = row[4]  # 5-й столбец

        if _norm(cell_num.value) == target_num and _norm(cell_date.value) == target_date:
            cell_reporting = ws_out.cell(row=cell_num.row, column=9, value=reporting_dates)
            cell_reporting.alignment = alignment
            cell_reporting.font = font
            found = True
            break

    wb_out.save(OUTPUT_PATH)
    wb_out.close()
    return found


def transfer_extracted_data_and_logic_to_excel(edu_loan_agr_num: str, edu_loan_agr_date: str, exctracted_fields: ExtractedFields) -> bool:
    """Заполняем графы 23-28 данными которые мы достали с помощью LLM и вывод в 29 граф"""
    if not OUTPUT_PATH.exists():
        if transfer_excel_data() != 0:
            print(f"File not found: {OUTPUT_PATH}")
            return False

    from datetime import date, datetime

    def _norm(value: object) -> str:
        if value is None:
            return ""
        if isinstance(value, (datetime, date)):
            return value.strftime("%Y-%m-%d")
        return str(value).strip()
    
    def _safe_value(value: object) -> str:
        if value is None:
            return "ОШИБКА"
        if isinstance(value, (datetime, date)):
            return value.strftime("%Y-%m-%d")
        text = str(value).strip()
        return text if text else "ОШИБКА"

    target_date = _norm(edu_loan_agr_date)
    target_num = _norm(edu_loan_agr_num)

    try:
        wb_out = load_workbook(OUTPUT_PATH)
    except TypeError as exc:
        if "extLst" in str(exc):
            if transfer_excel_data() != 0:
                print(f"Failed to rebuild output: {OUTPUT_PATH}")
                return False
            wb_out = load_workbook(OUTPUT_PATH)
        else:
            raise
    ws_out = wb_out.active

    alignment = Alignment(horizontal="center", vertical="center")
    font = Font(name="Times New Roman", size=12)

    found = False
    for row in ws_out.iter_rows(min_row=DATA_START_ROW):
        cell_num = row[3]  # 4-й столбец
        cell_date = row[4]  # 5-й столбец

        if _norm(cell_num.value) == target_num and _norm(cell_date.value) == target_date:
            ex_university = _safe_value(exctracted_fields.university_name)
            ex_student_fio = _safe_value(exctracted_fields.student_fio)
            ex_customer_fio = _safe_value(exctracted_fields.customer_fio)
            ex_paid_number = _safe_value(exctracted_fields.paid_edu_contract_number)
            ex_paid_date = _safe_value(exctracted_fields.paid_edu_contract_date)
            ex_specialty = _safe_value(exctracted_fields.specialty_code)

            extracted_values = [
                ex_university,
                ex_student_fio,
                ex_customer_fio,
                ex_paid_number,
                ex_paid_date,
                ex_specialty,
            ]

            for offset, value in enumerate(extracted_values, start=23):
                cell_reporting = ws_out.cell(row=cell_num.row, column=offset, value=value)
                cell_reporting.alignment = alignment
                cell_reporting.font = font


            # Тут мы формируем вывод ИИ по предоставленной логике
            cell_university_name = row[1].value
            cell_fio = row[2].value
            cell_cnp = row[5].value  # Код направления подготовки

            messages: list[str] = []

            cell_university_norm = _norm_text(str(cell_university_name) if cell_university_name is not None else "")
            ex_university_norm = _norm_text(ex_university)
            if cell_university_norm not in ("", "ОШИБКА") and ex_university_norm not in ("", "ОШИБКА"):
                if cell_university_norm != ex_university_norm:
                    messages.append("Иной вуз")

            cell_fio_norm = _norm_text(str(cell_fio) if cell_fio is not None else "")
            ex_fio_norm = _norm_text(ex_student_fio)
            if cell_fio_norm not in ("", "ОШИБКА") and ex_fio_norm not in ("", "ОШИБКА"):
                if cell_fio_norm != ex_fio_norm:
                    messages.append("Не найден")

                    cell_parts = cell_fio_norm.split()
                    ex_parts = ex_fio_norm.split()
                    if len(cell_parts) >= 2 and len(ex_parts) >= 2:
                        name_messages: list[str] = []
                        if cell_parts[0] != ex_parts[0]:
                            name_messages.append("Иная фамилия")
                        if cell_parts[1] != ex_parts[1]:
                            name_messages.append("Иное имя")
                        if name_messages:
                            if len(name_messages) == 2:
                                messages.append(" / ".join(name_messages))
                            else:
                                messages.append(name_messages[0])

            cell_cnp_norm = _norm_code(str(cell_cnp) if cell_cnp is not None else "")
            ex_specialty_norm = _norm_code(ex_specialty)
            if cell_cnp_norm not in ("", "ОШИБКА") and ex_specialty_norm not in ("", "ОШИБКА"):
                if cell_cnp_norm != ex_specialty_norm:
                    messages.append("Иное НПС")

            if "Иное НПС" in messages:
                messages.append(f"={ex_specialty}")

            result = "; ".join(messages)
            cell_reporting = ws_out.cell(row=cell_num.row, column=29, value=result)

            
            cell_reporting.alignment = alignment
            cell_reporting.font = font
            found = True
            break

    wb_out.save(OUTPUT_PATH)
    wb_out.close()
    return found

if __name__ == "__main__":
    # raise SystemExit(transfer_excel_data())
    transfer_reporting_data_to_excel("130291", "01.02.2025", "АВГУСТ;СЕНТЯБРЬ")
