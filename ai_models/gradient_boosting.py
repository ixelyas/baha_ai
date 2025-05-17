import os

# Путь к модели (если будешь использовать в будущем)
MODEL_PATH = os.path.join('models', 'gradient_boosting.pkl')

def predict(input_data):
    """
    input_data: list of [sleep, stress, fatigue, motivation, nutrition, soreness, mood]
    Возвращает уровень тренировки на основе вручную продуманной логики.
    """

    sleep, stress, fatigue, motivation, nutrition, soreness, mood = input_data

    # Имитируем сложную ансамблевую логику:
    score = (
        0.3 * sleep +
        -0.6 * stress +
        -0.5 * fatigue +
        1.0 * motivation +
        0.4 * nutrition +
        -0.4 * soreness +
        0.8 * mood
    )

    # Переводим score в категорию (1–5)
    if score >= 18:
        level = 5
    elif score >= 14:
        level = 4
    elif score >= 10:
        level = 3
    elif score >= 6:
        level = 2
    else:
        level = 1

    level_mapping = {
        1: "No Training",
        2: "Rest Day",
        3: "Light Training",
        4: "Moderate Training",
        5: "Go Hard"
    }

    return level_mapping[level]
