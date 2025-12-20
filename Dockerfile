# Базовий образ з Python
FROM python:3.10-slim

# Робоча директорія
WORKDIR /app

# Копіюємо requirements окремо (оптимізація кешу)
COPY requirements.txt .

# Встановлюємо залежності
RUN pip install --no-cache-dir -r requirements.txt

# Копіюємо весь код
COPY . .

# Streamlit порт
EXPOSE 8501

# Запуск Streamlit
CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501"]
