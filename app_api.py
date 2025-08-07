'''
Реализуем API с тремя эндпоинтами:

1. POST /predict —
Для получения предсказания от модели на основе входных данных

2. GET /stats —
Для получения статистики использования API

3. GET /health —
Для проверки работоспособности API

Шаг 1: Установка необходимых библиотек
pip install fastapi uvicorn pydantic scikit-learn pandas catboost

Шаг 2: Создание app_api.py
Шаг 3: Запуск приложения: python app_api.py
Шаг 4: Тестирование API

Тест API с помощью curl:

curl -X GET http://127.0.0.1:8000/health
curl -X GET http://127.0.0.1:8000/stats
curl -X POST http://127.0.0.1:8000/predict_model \
-H "Content-Type: application/json" \
-d '{"type": "Two year", "paperless_billing": "No", "payment_method": "Mailed check", "total_charges": 500.0, "senior_citizen": 1, "partner": "Yes", "dependents": "Yes", "internet_service": "No Internet", "online_security": "No Internet", "online_backup": "No Internet", "device_protection": "No Internet", "tech_support": "No Internet"}'

'''

from fastapi import FastAPI, Request, HTTPException
import pickle
import pandas as pd
from pydantic import BaseModel
import os

app = FastAPI()

model_path = os.path.join(os.path.dirname(__file__), 'best_catboost_model.pkl')
with open(model_path, 'rb') as f:
    best_catboost_model = pickle.load(f)

# Счетчик запросов
request_count = 0

# Модель для валидации входных данных
class PredictionInput(BaseModel):
    type: object
    paperless_billing: object
    payment_method: object
    total_charges: float
    senior_citizen: int
    partner: object 
    dependents: object
    internet_service: object
    online_security: object
    online_backup: object
    device_protection: object
    tech_support: object

@app.get("/stats")
def stats():
    return {"request_count": request_count}

@app.get("/health")
def health():
    return {"status": "OK"}

@app.post("/predict_model")
def predict_model(input_data: PredictionInput):
    global request_count
    global best_catboost_model

    request_count += 1

    new_data = pd.DataFrame({
        'type': [input_data.type],
        'paperless_billing': [input_data.paperless_billing],
        'payment_method': [input_data.payment_method],
        'total_charges': [input_data.total_charges],
        'senior_citizen': [input_data.senior_citizen],
        'partner': [input_data.partner], 
        'dependents':[input_data.dependents],
        'internet_service': [input_data.internet_service],
        'online_security': [input_data.online_security],
        'online_backup': [input_data.online_backup],
        'device_protection': [input_data.device_protection],
        'tech_support': [input_data.tech_support]
    })

    predictions = best_catboost_model.predict(new_data)
    result = "Contract Terminated" if predictions[0] == 1 else "Not Contract Terminated"

    return {"prediction": result}

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)



