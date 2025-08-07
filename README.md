# ML Deployment for Telecom Churn Prediction Using FastAPI & Docker

## Задача проекта

Проект направлен на предсказание оттока клиентов телеком-компании на основе исторических данных.

Обученная модель CatBoostClassifier обернута в FastAPI и упакована в Docker-контейнер для удобного локального запуска.

## План исследования

1. Загрузка библиотек и данных из базы
2. Исследовательский анализ и предобработка данных
3. Подготовка данных для обучения
4. Обучение моделей
5. Проверка качества лучшей модели на тестовой выборке
6. Построить Матрицу ошибок лучшей модели и график полноты и точности (precision recall curve)
7. Анализ важности входных признаков
8. Построить график зависимости самого важного фактора и целевой переменной
9. Вывод
10. Проверка модели на новых данных
11. Тест API

## Решение

Использована модель CatBoostClassifier
Метрика качества (ROC-AUC): 0.85
Accuracy: 0.80

## Инструкция по запуску проекта

1. Клонируйте репозиторий:

git clone https://github.com/elenadigital/Telecom_customer_churn_prediction_ML_deployment.git
cd Telecom_customer_churn_prediction_ML_deployment

2. Соберите образ Docker:

docker build -t churn_prediction_service:latest .

3. Запустите контейнер:

docker run -d --name churn_prediction_service -p 8000:8000 churn_prediction_service:latest

4. Проверьте API:

Откройте в браузере: http://localhost:8000/docs

**Доступные endpoints:**

* POST /predict — сделать предсказание

* GET /health — проверить, работает ли API

* GET /stats — получить статистику по числу запросов

## Используемые библиотеки и инструменты

Pandas, NumPy, Matplotlib, Seaborn, Scikitplot, Phik, Sklearn, Sqlalchemy, SQL, CatBoost, Pickle, FastAPI, Docker
