import os

# Путь к модели (если решишь обучать и использовать)
MODEL_PATH = os.path.join('models', 'random_forest.pkl')

def predict(input_data):
    """
    input_data: list of [sleep, stress, fatigue, motivation, nutrition, soreness, mood]
    Предсказание уровня тренировки на основе голосов нескольких "деревьев".
    """

    sleep, stress, fatigue, motivation, nutrition, soreness, mood = input_data

    votes = []

    # "Дерево 1"
    if motivation >= 4 and mood >= 4:
        votes.append(5)  # Go Hard
    elif fatigue <= 2:
        votes.append(3)  # Light Training
    else:
        votes.append(2)  # Rest Day

    # "Дерево 2"
    if stress >= 4 or soreness >= 4:
        votes.append(1)  # No Training
    elif sleep >= 3 and nutrition >= 3:
        votes.append(4)  # Moderate Training
    else:
        votes.append(2)  # Rest Day

    # "Дерево 3"
    if sleep <= 2 or fatigue >= 4:
        votes.append(1)  # No Training
    elif motivation >= 3:
        votes.append(4)  # Moderate Training
    else:
        votes.append(3)  # Light Training

    # Получаем наиболее часто встречающееся значение (как итог голосования)
    level = max(set(votes), key=votes.count)

    level_mapping = {
        1: "No Training",
        2: "Rest Day",
        3: "Light Training",
        4: "Moderate Training",
        5: "Go Hard"
    }

    return level_mapping[level]
