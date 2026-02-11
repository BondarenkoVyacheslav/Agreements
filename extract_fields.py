from pydantic import BaseModel, Field
from typing import Optional, List

class ExtractedFields(BaseModel):
    university_name: str = Field(description="Полное наименование ВУЗа/Исполнителя")
    student_fio: str = Field(description="ФИО обучающегося (полностью) или 'ОШИБКА'")
    customer_fio: str = Field(description="ФИО заказчика (полностью) или 'ОШИБКА'")
    paid_edu_contract_number: str = Field(description="Номер договора об оказании платных образовательных услуг или 'ОШИБКА'")
    paid_edu_contract_date: str = Field(description="Дата заключения договора (YYYY-MM-DD) или 'ОШИБКА'")
    specialty_code: str = Field(description="Код направления подготовки (например 09.03.03) или 'ОШИБКА'")

    # полезно для отладки/аудита:
    evidence: Optional[dict] = Field(default=None, description="Короткие цитаты/фрагменты, где найдено поле")