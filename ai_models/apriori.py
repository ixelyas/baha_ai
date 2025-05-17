# ai_models/apriori.py

def predict(input_data):
    """
    input_data: list of [sleep, stress, fatigue, motivation, nutrition, soreness, mood]
    """
    sleep, stress, fatigue, motivation, nutrition, soreness, mood = input_data

    # Простейшая логика на основе правил для моделирования Apriori
    score = 0

    # Положительные факторы
    if sleep >= 7:
        score += 1
    if motivation >= 7:
        score += 1
    if nutrition >= 7:
        score += 1
    if mood >= 7:
        score += 1

    # Отрицательные факторы
    if stress >= 7:
        score -= 1
    if fatigue >= 7:
        score -= 1
    if soreness >= 7:
        score -= 1

    # Определение результата на основе полученного балла
    if score >= 3:
        result = "Go Hard"
    elif score >= 1:
        result = "Moderate Training"
    elif score == 0:
        result = "Light Training"
    elif score == -1:
        result = "Rest Day"
    else:
        result = "No Training"

    return result
