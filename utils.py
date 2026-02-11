import re
from datetime import datetime

def _norm(s: str) -> str:
    if s is None:
        return ""
    s = s.strip().upper().replace("Ё", "Е")
    s = re.sub(r"\s+", " ", s)
    return s

def _norm_code(s: str) -> str:
    # 09.03.03 -> 09.03.03, "09 03 03" -> 09.03.03 (если вдруг)
    s = _norm(s)
    m = re.search(r"\b(\d{2})\D?(\d{2})\D?(\d{2})\b", s)
    return ".".join(m.groups()) if m else s

def _norm_date(s: str) -> str:
    s = _norm(s)
    if s == "ОШИБКА":
        return s
    # ожидаем YYYY-MM-DD; если пришло иначе — оставим как есть
    try:
        dt = datetime.strptime(s, "%Y-%m-%d")
        return dt.strftime("%Y-%m-%d")
    except Exception:
        return s

def metric(example, pred, trace=None) -> float:
    """DSPy metric: принимает (example, pred, trace=None)."""
    fields = [
        ("university_name", _norm),
        ("student_fio", _norm),
        ("customer_fio", _norm),
        ("paid_edu_contract_number", _norm),
        ("paid_edu_contract_date", _norm_date),
        ("specialty_code", _norm_code),
    ]
    ok = 0
    for f, norm_fn in fields:
        exp_v = norm_fn(getattr(example, f, ""))
        got_v = norm_fn(getattr(pred, f, ""))
        ok += int(exp_v == got_v)
    return ok / len(fields)
