import os

# Путь к модели (если будешь обучать настоящую модель)
MODEL_PATH = os.path.join('models', 'naive_bayes_5.pkl')

def predict(input_data):
    """
    input_data: list of [sleep, stress, fatigue, motivation, nutrition, soreness, mood]
    Имитация предсказания наивного Байеса вручную.
    """

    sleep, stress, fatigue, motivation, nutrition, soreness, mood = input_data

    # Условные вероятности: позитивные и негативные признаки
    positive = sum([
        1 if sleep >= 4 else 0,
        1 if motivation >= 4 else 0,
        1 if mood >= 4 else 0,
        1 if nutrition >= 4 else 0
    ])

    negative = sum([
        1 if stress >= 4 else 0,
        1 if fatigue >= 4 else 0,
        1 if soreness >= 4 else 0
    ])

    # Разница между «позитивными» и «негативными» вероятностями
    score = positive - negative

    # Оценка уровня тренировки на основе «вероятностной разницы»
    if score >= 3:
        level = 5
    elif score == 2:
        level = 4
    elif score == 1:
        level = 3
    elif score == 0:
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
