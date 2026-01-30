from __future__ import annotations

from contextlib import closing
from pathlib import Path

from openpyxl import load_workbook
from openpyxl.styles import Alignment, Font

SOURCE_PATH = Path("СВОД СБЕР июль-сентябрь.xlsx")
TEMPLATE_PATH = Path("Шаблон_выгрузки.xlsx")
OUTPUT_PATH = Path("СВОД_СБЕР_выгрузка.xlsx")
DATA_START_ROW = 2

def main() -> int:
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
                educational_loan_agreement_date,
                educational_loan_agreement_number,
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


if __name__ == "__main__":
    raise SystemExit(main())
