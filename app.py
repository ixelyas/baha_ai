import os
from werkzeug.utils import secure_filename
import sqlite3
import joblib
import numpy as np
import torch
import cv2
import pandas as pd
import spacy
from flask import (
    Flask,
    render_template,
    request,
    redirect,
    url_for,
    flash,
    jsonify,
    session,
)
from datetime import datetime
import ai_models.linear_regression as linear
import ai_models.logistic_regression as logistic
import ai_models.decision_tree as tree
import ai_models.random_forest as forest
import ai_models.naive_bayes as bayes
import ai_models.knn as knn
import ai_models.svm as svm
import ai_models.gradient_boosting as gb
import ai_models.kmeans as kmeans
import ai_models.apriori as apriori
import ai_models.pca as pca
import ai_models.yolo_detector as yolo

UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app = Flask(__name__)
app.secret_key = "svetlyiden_secret_key"

# Устанавливаем максимальный размер файла для загрузки
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB
app.config['UPLOAD_FOLDER'] = 'static/uploads'

# Путь для хранения загруженных файлов
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}


# Load spaCy model - fallback to simple tokenizer if loading fails
try:
    nlp = spacy.load('en_core_web_sm')
    print("Successfully loaded spaCy model")
except Exception as e:
    print(f"Error loading spaCy model: {e}")
    # Create a simple tokenizer as fallback
    from spacy.lang.en import English
    nlp = English()
    print("Using fallback English tokenizer")


def init_db():
    conn = sqlite3.connect("database.db")
    c = conn.cursor()

    c.execute("""
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """)

    c.execute("""
    CREATE TABLE IF NOT EXISTS survey_results (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER,
        sleep INTEGER,
        stress INTEGER,
        fatigue INTEGER,
        motivation INTEGER,
        nutrition INTEGER,
        soreness INTEGER,
        mood INTEGER,
        training_level INTEGER,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (user_id) REFERENCES users (id)
    )
    """)

    # 🟢 Добавить вот это (ЭТО У ТЕБЯ ПРОПУЩЕНО!!!)
    c.execute("""
    CREATE TABLE IF NOT EXISTS journal_entries (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER,
        content TEXT,
        sentiment_score REAL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (user_id) REFERENCES users (id)
    )
    """)

    conn.commit()
    conn.close()



init_db()

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def detect_objects(image_path):
    """
    Функция для детекции объектов на изображении с использованием модели YOLOv5.
    Возвращает список объектов и их вероятности.
    """
    # Загружаем модель YOLOv5
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s')  # маленькая версия YOLOv5

    # Читаем изображение
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Конвертируем BGR в RGB

    # Детекция объектов
    results = model(img)

    # Печатаем результаты
    results.print()  # Выводим объекты с их вероятностями

    # Получаем результаты
    detected_objects = results.names  # Список всех возможных объектов
    predictions = results.xywh[0].numpy()  # Координаты, вероятности и метки объектов

    return detected_objects, predictions

def get_user_id():
    if "user_id" not in session:
        conn = sqlite3.connect("database.db")
        c = conn.cursor()
        c.execute("INSERT INTO users (name) VALUES ('Anonymous')")
        user_id = c.lastrowid
        conn.commit()
        conn.close()
        session["user_id"] = user_id
    return session["user_id"]


def get_recommendations(training_level):
    if training_level == 1:
        return [
            "Strict Rest Day - avoid any physical activity.",
            "Focus on recovery: light stretching, deep breathing or meditation.",
            "Make sure to drink plenty of water and eat nourishing meals.",
            "Try to sleep well and avoid stress today."
        ]
    elif training_level == 2:
        return [
            "Rest Day - your body needs time to recover.",
            "Gentle mobility exercises or a casual walk are fine.",
            "Focus on high-protein meals to assist muscle recovery.",
            "Avoid heavy lifting or intense cardio."
        ]
    elif training_level == 3:
        return [
            "Light Training Day - keep the intensity low.",
            "Focus on cardio, yoga or bodyweight exercises.",
            "Stay aware of your energy levels and avoid overexertion.",
            "A balanced diet and hydration are still important today."
        ]
    elif training_level == 4:
        return [
            "Moderate Training Day - normal workout intensity is fine.",
            "Work on strength and endurance but avoid pushing to failure.",
            "Make sure to warm up properly before your workout.",
            "Stay hydrated and eat a good mix of protein and carbs."
        ]
    elif training_level == 5:
        return [
            "Go Hard - you are ready for an intense workout.",
            "Push yourself and aim for personal bests today.",
            "Ensure you warm up properly before starting.",
            "Eat enough carbs before and after your session for optimal performance."
        ]
    return ["No recommendation."]


def find_all(text, substring):
    """Find all occurrences of substring in text"""
    start = 0
    while True:
        start = text.find(substring, start)
        if start == -1:
            return
        yield start
        start += 1  # Move past the current match


def analyze_text(text):
    # Simple mood analysis focused on fitness-related expressions

    # Positive fitness-related words
    positive_words = [
        "ready",
        "motivated",
        "strong",
        "energetic",
        "powerful",
        "focused",
        "determined",
        "confident",
        "excited",
        "good",
        "great",
        "amazing",
        "fresh",
        "positive",
        "enthusiastic",
        "happy",
        "fit",
        "healthy",
        "go hard",
        "pumped",
        "recovered",
    ]

    # Negative fitness-related words
    negative_words = [
        "tired",
        "exhausted",
        "sore",
        "hurt",
        "injured",
        "pain",
        "weak",
        "lazy",
        "unmotivated",
        "bad",
        "sad",
        "down",
        "fatigued",
        "burnt out",
        "no energy",
        "stress",
        "stressed",
        "overtrained",
        "sluggish",
    ]

    positive_count = 0
    negative_count = 0

    # Convert text to lowercase for comparison
    text_lower = text.lower()

    # Count positive words
    for word in positive_words:
        positive_count += sum(1 for _ in find_all(text_lower, word))

    # Count negative words
    for word in negative_words:
        negative_count += sum(1 for _ in find_all(text_lower, word))

    # Calculate sentiment score (-1 to 1)
    total = positive_count + negative_count
    if total > 0:
        sentiment_score = (positive_count - negative_count) / total
    else:
        sentiment_score = 0

    print(
        f"Fitness Mood Analysis: Positive: {positive_count}, Negative: {negative_count}, Score: {sentiment_score}"
    )
    return sentiment_score


def get_fitness_recommendations(training_level, sentiment_score=None):
    recommendations = []

    if training_level == 1:  # No Training
        recommendations = [
            "Today is for strict rest — no workouts recommended.",
            "Focus on recovery activities like stretching or meditation.",
            "Ensure proper nutrition and hydration.",
        ]
    elif training_level == 2:  # Rest Day
        recommendations = [
            "Take a light walk or perform stretching.",
            "Avoid intense activities to allow recovery.",
            "Stay hydrated and eat well-balanced meals.",
        ]
    elif training_level == 3:  # Light Training
        recommendations = [
            "Consider light cardio or mobility exercises.",
            "Stay aware of any soreness during training.",
            "Avoid heavy lifting or intensive workouts.",
        ]
    elif training_level == 4:  # Moderate Training
        recommendations = [
            "You are ready for a balanced training session.",
            "Focus on technique and form today.",
            "Do not push to maximum intensity if feeling tired.",
        ]
    else:  # Go Hard
        recommendations = [
            "You are in great shape — push your limits today!",
            "Aim for personal records or high intensity workout.",
            "Don't forget proper warm-up and cool-down.",
        ]

    # Additional mood-based recommendations
    if sentiment_score is not None:
        if sentiment_score < -0.5:
            recommendations.append(
                "You seem low on motivation. Consider taking it easier or doing something enjoyable."
            )
        elif sentiment_score < 0:
            recommendations.append(
                "Your mood is a bit low. Try to start slow and see how you feel during training."
            )
        elif sentiment_score > 0.5:
            recommendations.append(
                "Great mood! This is the perfect day to go for your goals."
            )
        elif sentiment_score > 0:
            recommendations.append(
                "You are feeling good — maintain that positive energy in your workout."
            )

    return recommendations


def get_progress_data():
    user_id = get_user_id()
    conn = sqlite3.connect("database.db")
    conn.row_factory = sqlite3.Row
    c = conn.cursor()

    # Get survey results (training readiness levels)
    c.execute(
        "SELECT training_level, created_at FROM survey_results WHERE user_id = ? ORDER BY created_at",
        (user_id,),
    )
    survey_data = c.fetchall()

    # Get journal sentiment scores
    c.execute(
        "SELECT sentiment_score, created_at FROM journal_entries WHERE user_id = ? ORDER BY created_at",
        (user_id,),
    )
    journal_data = c.fetchall()

    conn.close()

    # Return empty data if no records found
    if not survey_data and not journal_data:
        return None

    
    # Format dates for display
    def format_date(date_str):
        try:
            # Handle bytes object
            if isinstance(date_str, bytes):
                date_str = date_str.decode("utf-8")

            dt = datetime.strptime(date_str, "%Y-%m-%d %H:%M:%S.%f")
        except ValueError:
            try:
                dt = datetime.strptime(date_str, "%Y-%m-%d %H:%M:%S")
            except ValueError:
                # If all else fails, return the string representation
                return str(date_str)
        return dt.strftime("%d.%m %H:%M")

    # Create data structure for Chart.js
    chart_data = {
        "labels": [],
        "datasets": [
            {
                "label": "Training Level (1-5)",
                "data": [],
                "borderColor": "#e74c3c",
                "backgroundColor": "rgba(231, 76, 60, 0.2)",
                "borderWidth": 2,
                "pointRadius": 4,
                "tension": 0.1,
                "yAxisID": "y-anxiety",
            },
            {
                "label": "Journal Mood (-1 to 1)",
                "data": [],
                "borderColor": "#3498db",
                "backgroundColor": "rgba(52, 152, 219, 0.2)",
                "borderWidth": 2,
                "pointRadius": 4,
                "tension": 0.1,
                "yAxisID": "y-mood",
            },
        ],
    }

    # Process survey data
    if survey_data:
        for row in survey_data:
            training_level = None
            try:
                if row["training_level"] is not None:
                    # Try to handle different types of data
                    if isinstance(row["training_level"], bytes):
                        # For binary data, try to extract a valid integer or default to None
                        try:
                            # First try to decode as string then convert to int
                            training_level = int(
                                row["training_level"].decode("utf-8").strip()
                            )
                        except (UnicodeDecodeError, ValueError):
                            # If it's not valid UTF-8 or not a number, try to interpret first byte as integer
                            if (
                                len(row["training_level"]) > 0
                                and 0 <= row["training_level"][0] <= 4
                            ):
                                training_level = row["anxiety_training_levellevel"][0]
                            else:
                                training_level = None
                    else:
                        # For non-binary data, convert directly
                        training_level = int(row["training_level"])
            except (ValueError, TypeError):
                # If conversion fails, use None
                training_level = None

            # Format date and ensure it's a string
            date = format_date(row["created_at"])

            # Only add valid data points
            if training_level is not None and 0 <= training_level <= 4:
                chart_data["labels"].append(date)
                chart_data["datasets"][0]["data"].append(training_level)

    # Process journal data - generate matching labels for consistent x-axis
    if journal_data:
        journal_dates = []
        journal_scores = []

        for row in journal_data:
            # Format the date
            date = format_date(row["created_at"])

            # Safely convert sentiment score to float
            sentiment_score = None
            try:
                if row["sentiment_score"] is not None:
                    if isinstance(row["sentiment_score"], bytes):
                        try:
                            sentiment_score = float(
                                row["sentiment_score"].decode("utf-8").strip()
                            )
                        except (UnicodeDecodeError, ValueError):
                            sentiment_score = None
                    else:
                        sentiment_score = float(row["sentiment_score"])
            except (ValueError, TypeError):
                sentiment_score = None

            # Only add valid data points
            if sentiment_score is not None and -1 <= sentiment_score <= 1:
                journal_dates.append(date)
                journal_scores.append(sentiment_score)

        # If we already have labels from survey data, align journal data with those labels
        if chart_data["labels"]:
            # Fill with null for dates that don't have journal entries
            for date in chart_data["labels"]:
                if date in journal_dates:
                    idx = journal_dates.index(date)
                    chart_data["datasets"][1]["data"].append(journal_scores[idx])
                else:
                    chart_data["datasets"][1]["data"].append(None)

            # Add any journal dates not already in labels
            for i, date in enumerate(journal_dates):
                if date not in chart_data["labels"]:
                    chart_data["labels"].append(date)
                    # Add null for anxiety data at this date
                    chart_data["datasets"][0]["data"].append(None)
                    chart_data["datasets"][1]["data"].append(journal_scores[i])
        else:
            # No survey data, just use journal dates as labels
            chart_data["labels"] = journal_dates
            chart_data["datasets"][1]["data"] = journal_scores

    # Sort all data by date
    if chart_data["labels"]:
        # Create tuples of (date, anxiety, mood) to sort by date
        combined = list(
            zip(
                chart_data["labels"],
                chart_data["datasets"][0]["data"],
                chart_data["datasets"][1]["data"],
            )
        )
        combined.sort(key=lambda x: x[0])

        # Unpack back into chart structure
        chart_data["labels"] = [item[0] for item in combined]
        chart_data["datasets"][0]["data"] = [item[1] for item in combined]
        chart_data["datasets"][1]["data"] = [item[2] for item in combined]

    return chart_data


# Routes
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/form", methods=["GET", "POST"])
def form():
    if request.method == "POST":
        # Сбор данных из формы
        sleep = int(request.form["sleep"])
        stress = int(request.form["stress"])
        fatigue = int(request.form["fatigue"])
        motivation = int(request.form["motivation"])
        nutrition = int(request.form["nutrition"])
        soreness = int(request.form["soreness"])
        mood = int(request.form["mood"])

        # Преобразование данных в массив numpy для предсказания
        features = np.array([[sleep, stress, fatigue, motivation, nutrition, soreness, mood]])

        # Загрузка модели Naive Bayes
        bayes = joblib.load('models/naive_bayes.pkl')

        # Прогноз уровня готовности к тренировке с использованием модели
        training_level = bayes.predict(features)[0]  # Модель возвращает массив, берем первый элемент

        # Сохраняем результаты в базу данных
        user_id = get_user_id()  # Получаем ID пользователя
        conn = sqlite3.connect("database.db")
        c = conn.cursor()
        c.execute(
            "INSERT INTO survey_results (user_id, sleep, stress, fatigue, motivation, nutrition, soreness, mood, training_level) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                user_id,
                sleep,
                stress,
                fatigue,
                motivation,
                nutrition,
                soreness,
                mood,
                training_level,
            ),
        )
        conn.commit()
        conn.close()

        # Получаем рекомендации на основе уровня тренированности
        recommendations = get_recommendations(training_level)

        return render_template('results.html', anxiety_level=training_level, recommendations=recommendations)

    return render_template("form.html")




@app.route("/ai-lab", methods=["GET", "POST"])
def ai_lab():
    results = {}
    summary = None
    chart_labels = []
    chart_data = []
    algo_chart_data = {}
    uploaded_image = None

    algos = {
        "Linear Regression": linear.predict,
        "Logistic Regression": logistic.predict,
        "Decision Tree": tree.predict,
        "Random Forest": forest.predict,
        "Naive Bayes": bayes.predict,
        "KNN": knn.predict,
        "SVM": svm.predict,
        "Gradient Boosting": gb.predict,
        "KMeans": kmeans.predict,
        "Apriori": apriori.predict,
        "PCA": pca.predict,
    }

    if request.method == "POST":
        sleep = int(request.form["sleep"])
        stress = int(request.form["stress"])
        fatigue = int(request.form["fatigue"])
        motivation = int(request.form["motivation"])
        nutrition = int(request.form["nutrition"])
        soreness = int(request.form["soreness"])
        mood = int(request.form["mood"])

        input_data = [sleep, stress, fatigue, motivation, nutrition, soreness, mood]

        selected_algos = request.form.getlist("selected_algos")
        action = request.form.get("action")

        if action == "compare" or not selected_algos:
            selected_algos = list(algos.keys())

        for name in selected_algos:
            prediction = algos[name](input_data)
            results[name] = prediction
            algo_chart_data[name] = extract_numeric_level(prediction)

        # YOLO Image
        if "yolo_image" in request.files:
            file = request.files["yolo_image"]
            if file and file.filename != "":
                file_path = f"static/uploads/{file.filename}"
                file.save(file_path)
                uploaded_image = file_path
                prediction = yolo.predict(file_path)  # Теперь вызывает predict
                results["YOLO (CV)"] = prediction
            else:
                results["YOLO (CV)"] = "No image"
        else:
            results["YOLO (CV)"] = "No image"

        # Prepare chart data
        chart_labels = []
        chart_data = []

        for name, value in results.items():
            numeric_value = extract_numeric_level(value)
            if numeric_value > 0:
                chart_labels.append(name)
                chart_data.append(numeric_value)

        # Calculate recommendation summary
        hard = sum(1 for res in chart_data if res >= 4)
        light = sum(1 for res in chart_data if res == 3)
        rest = sum(1 for res in chart_data if res <= 2)

        if hard >= 4:
            summary = "Recommended intensity: Go Hard"
        elif light >= 4:
            summary = "Recommended intensity: Light Training"
        else:
            summary = "Recommended intensity: Rest Day"

    return render_template(
        "ai_lab.html",
        results=results,
        summary=summary,
        chart_labels=chart_labels,
        chart_data=chart_data,
        algo_chart_data=algo_chart_data,
        uploaded_image=uploaded_image,
    )


def extract_numeric_level(prediction_text):
    # Проверка, является ли prediction_text строкой
    if isinstance(prediction_text, str):
        prediction_text = prediction_text.lower()
    elif isinstance(prediction_text, dict):
        # Если prediction_text — это словарь, например, результат YOLO
        prediction_text = str(prediction_text)  # Преобразуем в строку
    else:
        prediction_text = str(prediction_text)  # В любом другом случае приводим к строке

    # Логика определения уровня на основе текста
    if "no training" in prediction_text:
        return 1
    elif "rest day" in prediction_text:
        return 2
    elif "light training" in prediction_text:
        return 3
    elif "moderate training" in prediction_text:
        return 4
    elif "go hard" in prediction_text:
        return 5
    else:
        return 0  # fallback если вдруг непонятный результат



@app.route("/journal", methods=["GET", "POST"])
def journal():
    if request.method == "POST":
        content = request.form["content"]

        # Analyze text sentiment (mood analysis)
        sentiment_score = analyze_text(content)

        # Save to database
        user_id = get_user_id()
        conn = sqlite3.connect("database.db")
        c = conn.cursor()
        c.execute(
            "INSERT INTO journal_entries (user_id, content, sentiment_score) VALUES (?, ?, ?)",
            (user_id, content, sentiment_score),
        )
        conn.commit()
        conn.close()

        # Get the most recent training level for this user
        conn = sqlite3.connect("database.db")
        c = conn.cursor()
        c.execute(
            "SELECT training_level FROM survey_results WHERE user_id = ? ORDER BY created_at DESC LIMIT 1",
            (user_id,),
        )
        result = c.fetchone()
        conn.close()

        if result:
            training_level = result[0]
        else:
            # Default to Medium Training if no survey completed
            training_level = 3  # Moderate Training

        # Get recommendations based on training level and sentiment
        recommendations = get_fitness_recommendations(training_level, sentiment_score)

        # Determine mood text
        mood_text = "Neutral"
        if sentiment_score > 0.3:
            mood_text = "Positive"
        elif sentiment_score < -0.3:
            mood_text = "Negative"

        return render_template(
            "results.html",
            sentiment_score=sentiment_score,
            mood_text=mood_text,
            recommendations=recommendations,
        )

    return render_template("journal.html")


@app.route("/progress")
def progress():
    chart_data = get_progress_data()
    return render_template("progress.html", chart_data=chart_data)


@app.route("/api/progress-data")
def api_progress_data():
    """API endpoint to get progress data for Chart.js"""
    chart_data = get_progress_data()
    if chart_data:
        return jsonify(chart_data)
    return jsonify({"error": "No data available"}), 404



if __name__ == "__main__":
    app.run(debug=True)
