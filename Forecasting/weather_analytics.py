"""
Advanced Weather Analytics Module
==================================
This module provides sophisticated weather analysis algorithms including:
- Historical data analysis and trends
- Prediction accuracy tracking
- Weather pattern recognition
- Anomaly detection
- Statistical forecasting
- Extreme weather event detection
"""

import os
from datetime import datetime, timedelta

import joblib
import numpy as np
import pandas as pd
from django.conf import settings
from django.db.models import Avg, Count, Max, Min
from scipy import stats
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from .models import PredictionLog, SearchHistory, WeatherAlert


class WeatherAnalytics:
    """Advanced weather analytics and prediction analysis"""

    def __init__(self):
        self.models_dir = os.path.join(settings.BASE_DIR, "Forecasting", "models")

    def calculate_heat_index(self, temperature_c, humidity):
        """
        Calculate heat index (feels-like temperature) using advanced formula
        Returns temperature in Celsius
        """
        # Convert to Fahrenheit for calculation
        T = temperature_c * 9 / 5 + 32
        RH = humidity

        # Simple formula for low temperatures
        if T < 80:
            return temperature_c

        # Rothfusz regression for heat index
        HI = -42.379 + 2.04901523 * T + 10.14333127 * RH - 0.22475541 * T * RH
        HI += -0.00683783 * T * T - 0.05481717 * RH * RH + 0.00122874 * T * T * RH
        HI += 0.00085282 * T * RH * RH - 0.00000199 * T * T * RH * RH

        # Adjustments
        if RH < 13 and 80 <= T <= 112:
            adjustment = ((13 - RH) / 4) * np.sqrt((17 - abs(T - 95)) / 17)
            HI -= adjustment
        elif RH > 85 and 80 <= T <= 87:
            adjustment = ((RH - 85) / 10) * ((87 - T) / 5)
            HI += adjustment

        # Convert back to Celsius
        heat_index_c = (HI - 32) * 5 / 9
        return round(heat_index_c, 1)

    def calculate_wind_chill(self, temperature_c, wind_speed_kmh):
        """
        Calculate wind chill temperature
        Uses North American/UK formula
        """
        if temperature_c > 10 or wind_speed_kmh < 4.8:
            return temperature_c

        # Wind chill formula
        wind_chill = 13.12 + 0.6215 * temperature_c - 11.37 * (wind_speed_kmh**0.16)
        wind_chill += 0.3965 * temperature_c * (wind_speed_kmh**0.16)

        return round(wind_chill, 1)

    def calculate_dew_point(self, temperature_c, humidity):
        """
        Calculate dew point temperature using Magnus formula
        """
        a = 17.27
        b = 237.7

        alpha = ((a * temperature_c) / (b + temperature_c)) + np.log(humidity / 100.0)
        dew_point = (b * alpha) / (a - alpha)

        return round(dew_point, 1)

    def detect_weather_anomalies(self, current_data, historical_avg):
        """
        Detect anomalies in weather data using statistical methods
        Returns list of detected anomalies
        """
        anomalies = []
        z_threshold = 2.5  # 2.5 standard deviations

        for param in ["temperature", "humidity", "pressure", "wind_speed"]:
            if param in current_data and param in historical_avg:
                current = current_data[param]
                mean = historical_avg[param]["mean"]
                std = historical_avg[param]["std"]

                if std > 0:
                    z_score = abs((current - mean) / std)
                    if z_score > z_threshold:
                        anomalies.append(
                            {
                                "parameter": param,
                                "current_value": current,
                                "expected_mean": mean,
                                "z_score": round(z_score, 2),
                                "severity": "high" if z_score > 3 else "moderate",
                            }
                        )

        return anomalies

    def calculate_prediction_accuracy(self, city_name, days=7):
        """
        Calculate accuracy metrics for recent predictions
        Compares predictions with actual weather (if available)
        """
        from_date = datetime.now() - timedelta(days=days)
        predictions = PredictionLog.objects.filter(
            city_name=city_name, prediction_date__gte=from_date
        ).order_by("-prediction_date")

        if predictions.count() < 2:
            return None

        # Calculate metrics
        metrics = {
            "total_predictions": predictions.count(),
            "rain_predictions": predictions.filter(rain_prediction=1).count(),
            "avg_temp_predicted": [],
            "prediction_dates": [],
        }

        for pred in predictions[:10]:  # Last 10 predictions
            try:
                temps = (
                    eval(pred.temp_predictions)
                    if isinstance(pred.temp_predictions, str)
                    else pred.temp_predictions
                )
                if temps:
                    metrics["avg_temp_predicted"].append(np.mean(temps))
                    metrics["prediction_dates"].append(pred.prediction_date)
            except:
                pass

        return metrics

    def analyze_weather_trends(self, weather_data_list):
        """
        Analyze weather trends from historical data
        Returns trend analysis with predictions
        """
        if len(weather_data_list) < 3:
            return None

        df = pd.DataFrame(weather_data_list)
        trends = {}

        for column in ["temperature", "humidity", "pressure"]:
            if column in df.columns:
                values = df[column].values

                # Linear regression for trend
                x = np.arange(len(values))
                slope, intercept, r_value, p_value, std_err = stats.linregress(
                    x, values
                )

                # Determine trend direction
                if abs(slope) < 0.1:
                    direction = "stable"
                elif slope > 0:
                    direction = "increasing"
                else:
                    direction = "decreasing"

                trends[column] = {
                    "direction": direction,
                    "slope": round(slope, 3),
                    "r_squared": round(r_value**2, 3),
                    "forecast_next": round(intercept + slope * len(values), 1),
                }

        return trends

    def calculate_comfort_index(self, temperature, humidity, wind_speed):
        """
        Calculate comfort index based on multiple factors
        Returns score from 0-100 (100 being most comfortable)
        """
        score = 100

        # Temperature comfort (optimal: 20-24°C)
        if temperature < 10:
            score -= (10 - temperature) * 3
        elif temperature > 30:
            score -= (temperature - 30) * 3
        elif temperature < 20:
            score -= (20 - temperature) * 1.5
        elif temperature > 24:
            score -= (temperature - 24) * 1.5

        # Humidity comfort (optimal: 40-60%)
        if humidity < 30:
            score -= (30 - humidity) * 1
        elif humidity > 70:
            score -= (humidity - 70) * 1.5

        # Wind speed comfort (optimal: < 20 km/h)
        if wind_speed > 20:
            score -= (wind_speed - 20) * 0.5

        score = max(0, min(100, score))

        if score >= 80:
            level = "Excellent"
        elif score >= 60:
            level = "Good"
        elif score >= 40:
            level = "Fair"
        else:
            level = "Poor"

        return {"score": round(score, 1), "level": level}

    def detect_extreme_weather(self, weather_data):
        """
        Detect extreme weather conditions and return warnings
        """
        warnings = []

        temp = weather_data.get("current_temp", 0)
        humidity = weather_data.get("humidity", 0)
        wind_speed_kmh = weather_data.get("wind_speed", 0) * 3.6
        pressure = weather_data.get("pressure", 1013)

        # Extreme heat
        if temp > 40:
            warnings.append(
                {
                    "type": "extreme_heat",
                    "severity": "critical",
                    "message": f"Extreme heat warning: {temp}°C. Stay hydrated and avoid outdoor activities.",
                }
            )
        elif temp > 35:
            warnings.append(
                {
                    "type": "high_heat",
                    "severity": "warning",
                    "message": f"High temperature: {temp}°C. Take precautions in outdoor activities.",
                }
            )

        # Extreme cold
        if temp < -10:
            warnings.append(
                {
                    "type": "extreme_cold",
                    "severity": "critical",
                    "message": f"Extreme cold warning: {temp}°C. Risk of frostbite.",
                }
            )
        elif temp < 0:
            warnings.append(
                {
                    "type": "freezing",
                    "severity": "warning",
                    "message": f"Freezing temperatures: {temp}°C. Watch for ice.",
                }
            )

        # High winds
        if wind_speed_kmh > 75:
            warnings.append(
                {
                    "type": "high_wind",
                    "severity": "critical",
                    "message": f"Dangerous winds: {round(wind_speed_kmh, 1)} km/h. Avoid travel.",
                }
            )
        elif wind_speed_kmh > 50:
            warnings.append(
                {
                    "type": "strong_wind",
                    "severity": "warning",
                    "message": f"Strong winds: {round(wind_speed_kmh, 1)} km/h. Secure loose objects.",
                }
            )

        # Low pressure (storm indicator)
        if pressure < 980:
            warnings.append(
                {
                    "type": "low_pressure",
                    "severity": "warning",
                    "message": f"Very low pressure: {pressure} mb. Storm possible.",
                }
            )

        # Heat index danger
        heat_index = self.calculate_heat_index(temp, humidity)
        if heat_index > 40 and humidity > 60:
            warnings.append(
                {
                    "type": "heat_stress",
                    "severity": "warning",
                    "message": f"Heat stress risk. Feels like {heat_index}°C with {humidity}% humidity.",
                }
            )

        return warnings

    def calculate_precipitation_probability_advanced(
        self, humidity, pressure, temp, cloud_cover
    ):
        """
        Advanced precipitation probability using multiple factors
        Uses empirical rules and statistical correlations
        """
        prob = 0

        # Humidity factor (most important)
        if humidity > 80:
            prob += 40
        elif humidity > 70:
            prob += 25
        elif humidity > 60:
            prob += 10

        # Pressure factor (low pressure = higher chance)
        if pressure < 1000:
            prob += 25
        elif pressure < 1010:
            prob += 15
        elif pressure < 1013:
            prob += 5

        # Cloud cover factor
        if cloud_cover > 80:
            prob += 20
        elif cloud_cover > 60:
            prob += 10

        # Temperature factor (certain temps more conducive to rain)
        if 10 <= temp <= 25:
            prob += 5

        # Normalize to 0-100
        prob = min(100, prob)

        return round(prob, 1)

    def get_weather_recommendation(self, weather_data):
        """
        Generate personalized recommendations based on weather conditions
        """
        recommendations = []

        temp = weather_data.get("current_temp", 20)
        humidity = weather_data.get("humidity", 50)
        wind_speed_kmh = weather_data.get("wind_speed", 0) * 3.6
        rain_prediction = weather_data.get("rain_prediction", 0)
        description = weather_data.get("description", "").lower()

        # Clothing recommendations
        if temp < 5:
            recommendations.append(
                {
                    "category": "clothing",
                    "icon": "bi-snow",
                    "text": "Wear heavy winter clothing, gloves, and a warm hat.",
                }
            )
        elif temp < 15:
            recommendations.append(
                {
                    "category": "clothing",
                    "icon": "bi-thermometer-low",
                    "text": "Wear a jacket or sweater. Layering recommended.",
                }
            )
        elif temp > 30:
            recommendations.append(
                {
                    "category": "clothing",
                    "icon": "bi-thermometer-high",
                    "text": "Wear light, breathable clothing. Use sunscreen.",
                }
            )

        # Activity recommendations
        if rain_prediction == 1 or "rain" in description:
            recommendations.append(
                {
                    "category": "activity",
                    "icon": "bi-umbrella",
                    "text": "Carry an umbrella. Consider indoor activities.",
                }
            )
        elif 15 <= temp <= 25 and wind_speed_kmh < 20 and "clear" in description:
            recommendations.append(
                {
                    "category": "activity",
                    "icon": "bi-bicycle",
                    "text": "Perfect weather for outdoor activities!",
                }
            )

        # Health recommendations
        if humidity > 80 and temp > 25:
            recommendations.append(
                {
                    "category": "health",
                    "icon": "bi-droplet-fill",
                    "text": "High humidity. Stay hydrated and take breaks in AC.",
                }
            )
        elif temp > 35:
            recommendations.append(
                {
                    "category": "health",
                    "icon": "bi-exclamation-triangle",
                    "text": "Heat warning: Avoid prolonged sun exposure. Drink plenty of water.",
                }
            )

        # UV recommendations (simplified)
        if "clear" in description and 10 <= datetime.now().hour <= 16:
            recommendations.append(
                {
                    "category": "health",
                    "icon": "bi-brightness-high",
                    "text": "High UV exposure. Apply sunscreen and wear sunglasses.",
                }
            )

        return recommendations

    def compare_cities_weather(self, cities_data):
        """
        Compare weather conditions across multiple cities
        Returns comparative analysis
        """
        if len(cities_data) < 2:
            return None

        comparison = {
            "warmest": max(cities_data, key=lambda x: x["temp"]),
            "coldest": min(cities_data, key=lambda x: x["temp"]),
            "most_humid": max(cities_data, key=lambda x: x["humidity"]),
            "windiest": max(cities_data, key=lambda x: x["wind_speed"]),
            "average_temp": round(np.mean([c["temp"] for c in cities_data]), 1),
            "average_humidity": round(np.mean([c["humidity"] for c in cities_data]), 1),
        }

        return comparison

    def calculate_weather_volatility(self, historical_data):
        """
        Calculate weather volatility (how much weather changes)
        Higher volatility = less predictable weather
        """
        if len(historical_data) < 3:
            return None

        temps = [d["temp"] for d in historical_data]
        humidity = [d["humidity"] for d in historical_data]

        temp_volatility = np.std(temps)
        humidity_volatility = np.std(humidity)

        # Normalized volatility score (0-100)
        volatility_score = min(100, (temp_volatility * 2 + humidity_volatility / 2))

        if volatility_score < 20:
            stability = "Very Stable"
        elif volatility_score < 40:
            stability = "Stable"
        elif volatility_score < 60:
            stability = "Moderate"
        elif volatility_score < 80:
            stability = "Volatile"
        else:
            stability = "Highly Volatile"

        return {
            "score": round(volatility_score, 1),
            "stability": stability,
            "temp_std": round(temp_volatility, 2),
            "humidity_std": round(humidity_volatility, 2),
        }
