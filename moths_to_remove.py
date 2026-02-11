from datetime import date

REPORT_MONTHS = [
    (2025, 7, "ИЮЛЬ"),
    (2025, 8, "АВГУСТ"),
    (2025, 9, "СЕНТЯБРЬ"),
]

def months_to_remove(contract_dt: date) -> str:
    # берем все отчетные месяцы >= месяца договора
    # если договор раньше июля 2025 -> вернем все (ИЮЛЬ;АВГУСТ;СЕНТЯБРЬ)
    # если позже сентября 2025 -> вернем "" (или "ОШИБКА" — решите как вам нужно)
    res = []
    for y, m, label in REPORT_MONTHS:
        month_start = date(y, m, 1)
        if contract_dt <= month_start:
            res.append(label)
    # если договор попал внутрь отчетного месяца, тоже должен включать этот месяц:
    # например 2025-08-15 => АВГУСТ;СЕНТЯБРЬ
    # это покрывается условием contract_dt <= month_start? нет.
    # поэтому корректнее так:
    res = []
    for y, m, label in REPORT_MONTHS:
        if (contract_dt.year, contract_dt.month) <= (y, m):
            res.append(label)

    return "; ".join(res)
