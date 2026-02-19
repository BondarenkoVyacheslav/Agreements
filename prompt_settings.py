import dspy


ERROR_TOKEN = "ОШИБКА"


class SelectRelevantContractFragments(dspy.Signature):
    """Шаг 1: выдели только релевантные фрагменты для извлечения полей договора.

    Правила:
    1) Не пересказывай документ и не добавляй вымышленные данные.
    2) Возьми только короткие фрагменты, потенциально содержащие:
       - наименование вуза/исполнителя;
       - ФИО обучающегося;
       - ФИО заказчика;
       - номер договора;
       - дату договора;
       - код направления подготовки.
    3) Приоритетно отбирай фрагменты вокруг ключевых слов:
       "Договор №", "дата", "направление подготовки", "код направления подготовки",
       "Исполнитель", "Заказчик", "Обучающийся".
    4) Если есть несколько кандидатов, оставляй тот, что ближе к ключевым словам
       для нужного поля.
    5) Если релевантные фрагменты не найдены, верни ОШИБКА.
    """

    text: str = dspy.InputField(desc="Полный текст договора из txt-файла.")
    relevant_fragments: str = dspy.OutputField(
        desc="Только релевантные цитаты/строки без домыслов; иначе ОШИБКА."
    )


class StrictContractExtraction(dspy.Signature):
    """Шаг 2: извлеки 6 полей строго из `relevant_fragments`.

    Критические правила:
    1) Запрещено угадывать. Если поле не найдено или есть сомнение -> ОШИБКА.
    2) Не используй данные вне `relevant_fragments`.
    3) При множественных кандидатах выбирай тот, что ближе к ключевым словам:
       "Договор №", "дата", "направление подготовки".
    4) Ложные друзья:
       - номер договора не путать с номерами лицензий, приказов, паспортов,
         доверенностей, актов, регистрационных номеров;
       - дату договора не путать с датой рождения, датой выдачи документов,
         датой лицензии/приказа/доверенности.
    5) Формат даты договора строго YYYY-MM-DD, иначе ОШИБКА.
    6) specialty_code строго формата NN.NN.NN, иначе ОШИБКА.
    7) Для каждого поля верни короткий evidence-фрагмент (1-2 строки) из текста.
       Если поле равно ОШИБКА, evidence по этому полю тоже ОШИБКА.
    """

    relevant_fragments: str = dspy.InputField(
        desc="Релевантные фрагменты, полученные на Шаге 1."
    )
    university_name: str = dspy.OutputField(
        desc="Полное наименование ВУЗа/Исполнителя или ОШИБКА."
    )
    student_fio: str = dspy.OutputField(
        desc="ФИО обучающегося (полностью) или ОШИБКА."
    )
    customer_fio: str = dspy.OutputField(
        desc="ФИО заказчика (полностью) или ОШИБКА."
    )
    paid_edu_contract_number: str = dspy.OutputField(
        desc="Номер договора платных образовательных услуг или ОШИБКА."
    )
    paid_edu_contract_date: str = dspy.OutputField(
        desc="Дата договора строго YYYY-MM-DD или ОШИБКА."
    )
    specialty_code: str = dspy.OutputField(
        desc="Код направления строго NN.NN.NN или ОШИБКА."
    )
    evidence_university_name: str = dspy.OutputField(
        desc="Короткий фрагмент-основание для university_name или ОШИБКА."
    )
    evidence_student_fio: str = dspy.OutputField(
        desc="Короткий фрагмент-основание для student_fio или ОШИБКА."
    )
    evidence_customer_fio: str = dspy.OutputField(
        desc="Короткий фрагмент-основание для customer_fio или ОШИБКА."
    )
    evidence_paid_edu_contract_number: str = dspy.OutputField(
        desc="Короткий фрагмент-основание для paid_edu_contract_number или ОШИБКА."
    )
    evidence_paid_edu_contract_date: str = dspy.OutputField(
        desc="Короткий фрагмент-основание для paid_edu_contract_date или ОШИБКА."
    )
    evidence_specialty_code: str = dspy.OutputField(
        desc="Короткий фрагмент-основание для specialty_code или ОШИБКА."
    )


class TwoStageExtractor(dspy.Module):
    """Двухшаговый extraction: выборка релевантного текста -> строгое извлечение."""

    def __init__(self):
        super().__init__()
        self.select_relevant = dspy.Predict(SelectRelevantContractFragments)
        self.extract_strict = dspy.Predict(StrictContractExtraction)

    def forward(self, text: str):
        step1 = self.select_relevant(text=text)
        relevant_fragments = getattr(step1, "relevant_fragments", "") or ""

        extraction_input = text
        if relevant_fragments and str(relevant_fragments).strip().upper() != ERROR_TOKEN:
            extraction_input = relevant_fragments

        step2 = self.extract_strict(relevant_fragments=extraction_input)
        setattr(step2, "relevant_fragments", relevant_fragments)
        return step2
