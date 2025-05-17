import sqlite3

# Подключение к базе данных
conn = sqlite3.connect("database.db")
conn.row_factory = sqlite3.Row  # Это позволяет обращаться к строкам как к словарям
c = conn.cursor()

# Выполнение SQL-запроса
c.execute("SELECT * FROM survey_results")  # Или другой ваш запрос

# Получаем все данные
rows = c.fetchall()

# Печатаем данные
for row in rows:
    print(dict(row))  # Печатаем каждую запись как словарь

# Закрытие соединения
conn.close()
