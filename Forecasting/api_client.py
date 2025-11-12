import os
from datetime import datetime

import joblib
import pandas as pd
import pytz
import requests
from django.conf import settings

API_KEY = settings.OPENWEATHER_API_KEY
BASE_URL = settings.OPENWEATHER_BASE_URL
MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")


def get_current_weather(city):
    url = f"{BASE_URL}weather?q={city}&appid={API_KEY}&units=metric"
    try:
        response = requests.get(url)
        data = response.json()
        if response.status_code != 200 or data.get("cod") == "404":
            print(
                f"API returned an error or city not found: {data.get('message', 'Status code was not 200')}"
            )
            return None

        if data.get("cod") != 200:
            raise ValueError(f"API error: {data.get('message', 'Unknown error')}")

        utc_dt = datetime.fromtimestamp(data["dt"], tz=pytz.UTC)
        api_timestamp = utc_dt.strftime("%Y-%m-%d %H:%M:%S UTC")
        print(f"API Data Timestamp: {api_timestamp}")

        weather_data = {
            "city": data["name"],
            "current_temp": float(data["main"].get("temp", 0)),
            "feels_like": round(float(data["main"].get("feels_like", 0))),
            "temp_min": round(float(data["main"].get("temp_min", 0))),
            "temp_max": round(float(data["main"].get("temp_max", 0))),
            "humidity": round(float(data["main"].get("humidity", 0))),
            "description": data["weather"][0].get("description", "unknown"),
            "country": data["sys"].get("country", "Unknown"),
            "wind_dir": float(data["wind"].get("deg", 0)),
            "wind_speed": float(data["wind"].get("speed", 0)),
            "pressure": float(data["main"].get("pressure", 0)),
            "rainfall": data.get("rain", {}).get("1h", 0),
            "clouds": int(data["clouds"].get("all")),
            "visibility": int(data.get("visibility", 0)),
            "timezone_offset": int(data.get("timezone", 0)),
        }

        last_row = pd.DataFrame(
            [
                {
                    "Temp": weather_data["current_temp"],
                    "MinTemp": weather_data["temp_min"],
                    "MaxTemp": weather_data["temp_max"],
                    "Humidity": weather_data["humidity"],
                    "Pressure": weather_data["pressure"],
                    "WindSpeed": weather_data["wind_speed"] * 3.6,
                    "WindDir": weather_data["wind_dir"],
                    "Rainfall": weather_data["rainfall"],
                    "RainToday": None,
                }
            ]
        )
        os.makedirs(MODELS_DIR, exist_ok=True)
        joblib.dump(last_row, os.path.join(MODELS_DIR, "last_day.joblib"))

        print(f"Successfully fetched API Data for {city}.")
        return weather_data

    except (requests.RequestException, ValueError, KeyError) as e:
        print(f"API Request Failed for {city}: {e}. Using fallback data.")
        try:
            last_row = joblib.load(os.path.join(MODELS_DIR, "last_day.joblib")).iloc[0]
            print("Using fallback data from the last day of the dataset.")
            return {
                "city": city,
                "current_temp": round(float(last_row["Temp"])),
                "feels_like": round(float(last_row["Temp"])),
                "temp_min": round(float(last_row["MinTemp"])),
                "temp_max": round(float(last_row["MaxTemp"])),
                "humidity": round(float(last_row["Humidity"])),
                "description": "Data unavailable (fallback)",
                "country": "Unknown",
                "wind_dir": float(last_row["WindDir"]),
                "wind_speed": float(last_row["WindSpeed"]) / 3.6,
                "pressure": float(last_row["Pressure"]),
                "rainfall": float(last_row["Rainfall"]),
                "clouds": 50,
                "visibility": 10000,
                "timezone_offset": 0,
            }
        except Exception as fallback_e:
            print(f"Fallback data error: {fallback_e}")
            return None
