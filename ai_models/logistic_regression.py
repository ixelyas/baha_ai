import os
import math

# Путь к модели (если будешь использовать обученную)
MODEL_PATH = os.path.join('models', 'logistic_regression.pkl')

def sigmoid(z):
    return 1 / (1 + math.exp(-z))

def predict(input_data):
    """
    input_data: list of [sleep, stress, fatigue, motivation, nutrition, soreness, mood]
    Реализована логистическая регрессия вручную.
    """

    # Весовые коэффициенты (подобраны вручную, могут быть обучены)
    weights = [0.5, -0.9, -0.6, 1.2, 0.4, -0.5, 1.0]
    bias = -1.0

    z = sum(w * x for w, x in zip(weights, input_data)) + bias
    prob = sigmoid(z)

    # Преобразуем вероятность в категорию 1–5
    if prob > 0.85:
        level = 5
    elif prob > 0.65:
        level = 4
    elif prob > 0.45:
        level = 3
    elif prob > 0.25:
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
