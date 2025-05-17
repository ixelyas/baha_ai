from deepface import DeepFace

def analyze_emotion(image_path):
    try:
        result = DeepFace.analyze(img_path=image_path, actions=['emotion'], enforce_detection=False)
        dominant_emotion = result[0]['dominant_emotion']
        score = result[0]['emotion'][dominant_emotion]
        return dominant_emotion, round(score / 100, 2)
    except Exception as e:
        print("Ошибка анализа:", e)
        return None, 0.0

def emotion_to_training_level(emotion):
    """
    Карта эмоций -> рекомендаций по тренировке
    """
    mapping = {
        "happy": (5, "Go Hard"),
        "surprise": (4, "Moderate Training"),
        "neutral": (3, "Light Training"),
        "sad": (2, "Rest Day"),
        "angry": (2, "Rest Day"),
        "fear": (1, "No Training"),
        "disgust": (1, "No Training"),
        "calm": (3, "Light Training"),
        "tired": (1, "No Training")
    }
    return mapping.get(emotion, (2, "Rest Day"))

# <<< Эта функция должна называться predict для Flask >>>
def predict(image_path):
    emotion, confidence = analyze_emotion(image_path)
    if not emotion:
        return {
            "level": 1,
            "label": "No Training",
            "emotion": "Unknown",
            "confidence": 0.0
        }

    level, label = emotion_to_training_level(emotion)
    return {
        "level": level,
        "label": label,
        "emotion": emotion,
        "confidence": confidence
    }
