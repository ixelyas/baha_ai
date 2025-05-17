import os

# Путь к модели (если нужна)
MODEL_PATH = os.path.join('models', 'svm.pkl')

def predict(input_data):
    """
    input_data: list of [sleep, stress, fatigue, motivation, nutrition, soreness, mood]
    Реализовано вручную приближённое поведение линейного SVM.
    """

    # Весовые коэффициенты и смещение (подобраны вручную)
    weights = [0.7, -1.0, -0.8, 1.5, 0.5, -0.6, 1.0]
    bias = -3.5

    # Линейная функция разделения (гиперплоскость)
    decision_value = sum(w * x for w, x in zip(weights, input_data)) + bias

    # Имитация margin-based классификации
    if decision_value >= 4.0:
        level = 5  # Go Hard
    elif decision_value >= 2.0:
        level = 4  # Moderate Training
    elif decision_value >= 0.5:
        level = 3  # Light Training
    elif decision_value >= -1.0:
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
