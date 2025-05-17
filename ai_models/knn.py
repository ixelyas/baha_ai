import os

# Путь к модели (если потребуется)
MODEL_PATH = os.path.join('models', 'knn.pkl')

def predict(input_data):
    """
    input_data: list of [sleep, stress, fatigue, motivation, nutrition, soreness, mood]
    Возвращает рекомендацию на основе вручную имитированного KNN-подхода.
    """

    sleep, stress, fatigue, motivation, nutrition, soreness, mood = input_data

    # Простейший аналог KNN — сумма по позитивным и негативным признакам
    positive_factors = sleep + motivation + mood + nutrition
    negative_factors = stress + fatigue + soreness

    score = positive_factors - negative_factors

    # Пороговая логика, имитирующая голосование соседей
    if score >= 8:
        level = 5  # Go Hard
    elif score >= 4:
        level = 4  # Moderate Training
    elif score >= 1:
        level = 3  # Light Training
    elif score >= -2:
        level = 2  # Rest Day
    else:
        level = 1  # No Training

    level_mapping = {
        1: "No Training",
        2: "Rest Day",
        3: "Light Training",
        4: "Moderate Training",
        5: "Go Hard"
    }

    return level_mapping[level]
