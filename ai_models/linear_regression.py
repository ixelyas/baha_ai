import os

# Путь к модели (если нужно использовать обученную)
MODEL_PATH = os.path.join('models', 'linear_regression.pkl')

def predict(input_data):
    """
    input_data: list of [sleep, stress, fatigue, motivation, nutrition, soreness, mood]
    Выполняет предсказание вручную по формуле линейной регрессии.
    """

    # Коэффициенты (weights) и bias (сдвиг) — придуманы вручную
    weights = [0.6, -0.8, -0.7, 1.0, 0.5, -0.4, 0.9]
    bias = 1.5

    # Линейная регрессия: y = w1*x1 + w2*x2 + ... + wn*xn + b
    prediction = sum(w * x for w, x in zip(weights, input_data)) + bias

    # Округляем и ограничиваем в диапазоне [1, 5]
    level = int(min(5, max(1, round(prediction))))

    level_mapping = {
        1: "No Training",
        2: "Rest Day",
        3: "Light Training",
        4: "Moderate Training",
        5: "Go Hard"
    }

    return level_mapping[level]
