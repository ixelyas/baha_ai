import os

# Путь к модели, если захочешь её обучать и использовать
MODEL_PATH = os.path.join('models', 'kmeans.pkl')

def predict(input_data):
    """
    input_data: list of [sleep, stress, fatigue, motivation, nutrition, soreness, mood]
    Возвращает рекомендацию на основе вручную имитированной кластеризации.
    """

    sleep, stress, fatigue, motivation, nutrition, soreness, mood = input_data

    # Простая имитация кластеризации:
    if motivation >= 4 and mood >= 4 and fatigue <= 2:
        cluster = 2  # Go Hard
    elif stress >= 4 and soreness >= 4:
        cluster = 1  # Rest Day
    elif fatigue >= 4 or sleep <= 2:
        cluster = 4  # No Training
    elif nutrition >= 3 and sleep >= 3:
        cluster = 3  # Moderate Training
    else:
        cluster = 0  # Light Training

    cluster_mapping = {
        0: "Light Training",
        1: "Rest Day",
        2: "Go Hard",
        3: "Moderate Training",
        4: "No Training"
    }

    return cluster_mapping[cluster]
