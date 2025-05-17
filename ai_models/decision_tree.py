import os

# Если тебе всё же нужно обучать и сохранять модель — оставь это в другом скрипте (trainer.py)
MODEL_PATH = os.path.join('models', 'decision_tree.pkl')

def predict(input_data):
    """
    input_data: list of [sleep, stress, fatigue, motivation, nutrition, soreness, mood]
    Возвращает текстовую рекомендацию уровня тренировки.
    """

    sleep, stress, fatigue, motivation, nutrition, soreness, mood = input_data

    # Эмулируем "дерево решений" вручную, как будто оно обучено
    if stress >= 4 and fatigue >= 4:
        level = 1
    elif soreness >= 4 or sleep <= 2:
        level = 2
    elif motivation >= 4 and mood >= 4:
        level = 5
    elif motivation >= 3 and nutrition >= 3:
        level = 4
    elif fatigue <= 2 and sleep >= 3:
        level = 3
    else:
        level = 2

    level_mapping = {
        1: "No Training",
        2: "Rest Day",
        3: "Light Training",
        4: "Moderate Training",
        5: "Go Hard"
    }

    return level_mapping[level]
