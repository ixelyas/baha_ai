# ai_models/pca.py

def predict(input_data):
    """
    input_data: list of [sleep, stress, fatigue, motivation, nutrition, soreness, mood]
    """

    sleep, stress, fatigue, motivation, nutrition, soreness, mood = input_data

    # Simple sum of values to simulate PCA clusters
    total_score = sleep + (10 - stress) + (10 - fatigue) + motivation + nutrition + (10 - soreness) + mood

    # Classify based on total_score
    if total_score <= 14:
        result = "No Training"
    elif total_score <= 21:
        result = "Rest Day"
    elif total_score <= 28:
        result = "Light Training"
    elif total_score <= 35:
        result = "Moderate Training"
    else:
        result = "Go Hard"

    return result
