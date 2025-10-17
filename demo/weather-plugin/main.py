from pydantic import BaseModel
from fastapi import FastAPI

#!pip install fastapi uvicorn

# 创建一个FastAPI应用
# uvicorn main:app --reload --port 5002
app = FastAPI()

class WeatherResponse(BaseModel):
    city: str
    temperature: float
    condition: str
    humidity: int

# http://127.0.0.1:5002/weather/today?city=%E5%8C%97%E4%BA%AC
@app.get("/weather/today", response_model=WeatherResponse)
async def get_weather(city: str):
    # 这里应接入真实天气API（如OpenWeatherMap）
    # 以下是模拟数据示例
    return {
        "city": city,
        "temperature": 25.5,
        "condition": "晴",
        "humidity": 60
    }