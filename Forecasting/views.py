
from django.shortcuts import render, redirect
from django.contrib import messages
from django.conf import settings
import os
import joblib
import pytz
import pandas as pd
from datetime import datetime, timedelta
from .predict_future import predict_future
from .api_client import get_current_weather

MODELS_DIR = os.path.join(settings.BASE_DIR, 'Forecasting', 'models')
MODELS_LOADED = False
try:
    rain_model = joblib.load(os.path.join(MODELS_DIR, 'rain_model.joblib'))
    temp_model = joblib.load(os.path.join(MODELS_DIR, 'temp_model.joblib'))
    humidity_model = joblib.load(os.path.join(MODELS_DIR, 'humidity_model.joblib'))
    features_loaded = joblib.load(os.path.join(MODELS_DIR, 'features.joblib'))
    MODELS_LOADED = True
    print("Models loaded successfully at startup.")
except FileNotFoundError:
    rain_model, temp_model, humidity_model, features_loaded = None, None, None, None
    print("MODELS NOT FOUND. Please train the models first.")

def map_weather_to_class(desc):
    desc = desc.lower()
    if "clear" in desc:
        return "clear"
    if "cloud" in desc:
        return "clouds"
    if "overcast" in desc:
        return "overcast"
    if "rain" in desc:
        return "rain"
    if "snow" in desc:
        return "snow"
    if "drizzle" in desc:
        return "drizzle"
    if "thunder" in desc:
        return "thunder"
    if "mist" in desc or "haze" in desc or "smoke" in desc or "fog" in desc:
        return "mist"
    return "default"


def index_view(request):
    context = {}
    if messages.get_messages(request):
        context['messages'] = messages.get_messages(request)
        print("Messages in index_view:", [str(m) for m in messages.get_messages(request)])
    return render(request, 'index.html', context)

def weather_view(request):
    if request.method != 'POST':
        return redirect('index_view')
    
    if not MODELS_LOADED:
        messages.error(request, "Prediction models are not ready. Please run the training script.")
        return redirect('index_view')

    city = request.POST.get('city', '').strip()
    if not city:
        messages.error(request, "Error: City name cannot be empty.")
        return redirect('index_view')
    
    current_weather = get_current_weather(city)
    if current_weather is None:
        messages.error(request, f"Couldn't find city named '{city}'. Please enter a valid city.")
        return render(request, 'index.html')

    tz_offset = timedelta(seconds=current_weather.get('timezone_offset', 0))
    city_timezone = pytz.FixedOffset(tz_offset.total_seconds() / 60)
    now = datetime.now(city_timezone)
    next_hour = now.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)
    future_times = [(next_hour + timedelta(hours=i)).strftime("%H:00") for i in range(5)]
    
    current_features = {
        'MinTemp': current_weather['temp_min'],
        'MaxTemp': current_weather['temp_max'],
        'Rainfall': current_weather.get('rainfall', 0),
        'WindDir': current_weather['wind_dir'],
        'Pressure': current_weather['pressure'],
        'WindSpeed': current_weather['wind_speed'] * 3.6,
        'Temp': current_weather['current_temp'],
        'Humidity': current_weather['humidity'],
        'day_of_year': now.timetuple().tm_yday,
        'month': now.month,
        'day_of_week': now.weekday()
    }

    try:
        rain_features = features_loaded.get('rain', [])
        temp_features = features_loaded.get('temp', [])
        humidity_features = features_loaded.get('humidity', [])
        if not all([rain_features, temp_features, humidity_features]):
            raise ValueError("One or more feature lists in features_loaded are empty or missing")

        current_df_rain = pd.DataFrame([current_features])[rain_features].fillna(0)
        rain_prediction = rain_model.predict(current_df_rain.values)[0]
        future_temp_preds = predict_future(temp_model, current_features, temp_features, start_time=now)
        future_humidity_preds = predict_future(humidity_model, current_features, humidity_features, start_time=now)
        context = {
            'city': current_weather['city'],
            'country': current_weather['country'],
            'date': now.strftime('%Y-%m-%d'),
            'time': now.strftime('%H:%M %p'),
            'current_temp': current_weather['current_temp'],
            'feels_like': current_weather['feels_like'],
            'clouds': current_weather['clouds'],
            'humidity': current_weather['humidity'],
            'visibility': current_weather['visibility']/1000,
            'wind_speed': round(current_weather['wind_speed'], 1),
            'wind_dir': current_weather['wind_dir'],
            'pressure': current_weather['pressure'],
            'rain_prediction': 'a chance of rain' if rain_prediction == 1 else 'no chance of rain',
            'description': current_weather['description'].lower(),
            'main_class': map_weather_to_class(current_weather['description']),
            'time1': future_times[0],
            'temp1': round(future_temp_preds[0], 1),
            'hum1': round(future_humidity_preds[0], 1),
            'time2': future_times[1],
            'temp2': round(future_temp_preds[1], 1),
            'hum2': round(future_humidity_preds[1], 1),
            'time3': future_times[2],
            'temp3': round(future_temp_preds[2], 1),
            'hum3': round(future_humidity_preds[2], 1),
            'time4': future_times[3],
            'temp4': round(future_temp_preds[3], 1),
            'hum4': round(future_humidity_preds[3], 1),
            'time5': future_times[4],
            'temp5': round(future_temp_preds[4], 1),
            'hum5': round(future_humidity_preds[4], 1),
        }
        return render(request, 'weather.html', context)
    except Exception as e:
        print(f"Error during prediction: {type(e).__name__}: {str(e)}")
        messages.error(request, f"Prediction failed: {str(e)}")
        return redirect('index_view')