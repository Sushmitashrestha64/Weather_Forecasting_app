"""
Intelligent Weather Insights & Actionable Suggestions
=====================================================
This module provides meaningful, actionable insights based on weather patterns:
- Optimal time recommendations for activities
- Travel advisories
- Severe weather preparedness tips
- Historical pattern matching
"""

from datetime import datetime, timedelta

import numpy as np
from django.db.models import Avg, Max, Min

from .models import PredictionLog, SearchHistory


class WeatherInsights:
    """Generate intelligent, actionable weather insights"""

    def get_comprehensive_insights(self, weather_data, forecast_data, city):
        """
        Generate comprehensive insights package
        Returns actionable suggestions across multiple categories
        """
        insights = {
            "travel": self.get_travel_insights(weather_data),
            "best_times": self.get_optimal_times(forecast_data),
            "alerts": self.get_smart_alerts(weather_data, city),
            "comparison": self.get_historical_comparison(city, weather_data),
        }
        return insights

    def get_travel_insights(self, weather_data):
        """
        Travel advisories and recommendations
        """
        temp = weather_data.get("current_temp", 20)
        description = weather_data.get("description", "").lower()
        wind_speed_kmh = weather_data.get("wind_speed", 0) * 3.6
        visibility_km = weather_data.get("visibility", 10000) / 1000

        travel_insights = {
            "overall_conditions": "good",
            "driving_conditions": [],
            "flight_considerations": [],
            "public_transport": [],
            "recommendations": [],
        }

        # Visibility-based advisories
        if visibility_km < 1:
            travel_insights["overall_conditions"] = "poor"
            travel_insights["driving_conditions"].append(
                {
                    "severity": "high",
                    "issue": "Very low visibility (<1 km)",
                    "advice": "Avoid driving if possible. Use fog lights, reduce speed significantly.",
                }
            )
        elif visibility_km < 5:
            travel_insights["overall_conditions"] = "fair"
            travel_insights["driving_conditions"].append(
                {
                    "severity": "moderate",
                    "issue": "Reduced visibility",
                    "advice": "Drive carefully, use low beams, increase following distance.",
                }
            )

        # Weather condition advisories
        if "rain" in description or "drizzle" in description:
            travel_insights["driving_conditions"].append(
                {
                    "severity": "moderate",
                    "issue": "Wet roads",
                    "advice": "Reduce speed by 25%, increase braking distance, avoid sudden maneuvers.",
                }
            )
            travel_insights["public_transport"].append(
                "Expect delays. Roads may be congested. Plan extra 15-30 minutes."
            )

        if "thunderstorm" in description or "thunder" in description:
            travel_insights["overall_conditions"] = "poor"
            travel_insights["flight_considerations"].append(
                "⚠️ Flights may be delayed or cancelled due to thunderstorms. Check with airline."
            )
            travel_insights["recommendations"].append(
                "Delay non-essential travel. If driving, pull over safely during heavy storms."
            )

        if "snow" in description:
            travel_insights["overall_conditions"] = "poor"
            travel_insights["driving_conditions"].append(
                {
                    "severity": "high",
                    "issue": "Snow on roads",
                    "advice": "Use winter tires. Drive at least 50% slower. Keep emergency kit in car.",
                }
            )

        if wind_speed_kmh > 50:
            travel_insights["driving_conditions"].append(
                {
                    "severity": "moderate",
                    "issue": "Strong crosswinds",
                    "advice": "Hold steering wheel firmly, reduce speed on bridges and open areas.",
                }
            )
            travel_insights["flight_considerations"].append(
                "Expect turbulence. Flights may be delayed."
            )

        # Temperature-based advisories
        if temp < -5:
            travel_insights["driving_conditions"].append(
                {
                    "severity": "moderate",
                    "issue": "Risk of ice on roads",
                    "advice": "Watch for black ice especially on bridges. Start car earlier to warm up.",
                }
            )

        # Good conditions
        if (
            15 <= temp <= 28
            and visibility_km > 10
            and "rain" not in description
            and wind_speed_kmh < 20
        ):
            travel_insights["overall_conditions"] = "excellent"
            travel_insights["recommendations"].append(
                "✓ Excellent conditions for travel. Consider taking scenic routes!"
            )

        return travel_insights

    def get_optimal_times(self, forecast_data):
        """
        Determine best times for various activities based on forecast
        """
        if not forecast_data:
            return None

        optimal_times = {
            "exercise": None,
            "outdoor_dining": None,
            "commute": None,
            "laundry": None,
            "car_wash": None,
        }

        # Analyze forecast (simplified - assumes forecast is list of hourly data)
        # Find coolest hour for exercise
        # Find clearest weather for outdoor activities
        # etc.

        optimal_times["exercise"] = {
            "recommended": "Early morning (6-8 AM) or evening (5-7 PM)",
            "reason": "Cooler temperatures, less UV exposure",
        }

        optimal_times["outdoor_dining"] = {
            "recommended": "Evening (6-8 PM)",
            "reason": "Pleasant temperature, less direct sunlight",
        }

        return optimal_times

    def get_smart_alerts(self, weather_data, city):
        """
        Intelligent, context-aware alerts
        """
        alerts = []

        temp = weather_data.get("current_temp", 20)
        description = weather_data.get("description", "").lower()
        pressure = weather_data.get("pressure", 1013)

        # Rapid pressure drop = storm coming
        if pressure < 995:
            alerts.append(
                {
                    "priority": "high",
                    "icon": "⛈️",
                    "title": "Storm Warning",
                    "message": "Very low pressure detected. Storm likely within 12-24 hours.",
                    "actions": [
                        "Secure outdoor items",
                        "Charge devices",
                        "Check emergency supplies",
                    ],
                }
            )

        # Temperature anomaly
        if temp > 35:
            alerts.append(
                {
                    "priority": "high",
                    "icon": "🌡️",
                    "title": "Extreme Heat",
                    "message": f"Temperature is {temp}°C. Heat exhaustion risk is high.",
                    "actions": [
                        "Stay hydrated",
                        "Avoid outdoor activities",
                        "Check on vulnerable people",
                    ],
                }
            )

        # Air quality (if implemented)
        # Pollen levels (if implemented)
        # UV index warnings

        return alerts

    def get_historical_comparison(self, city, current_weather):
        """
        Compare current weather to historical patterns
        """
        try:
            # Get historical predictions for this city
            month = datetime.now().month
            historical_logs = PredictionLog.objects.filter(
                city_name__iexact=city, prediction_date__month=month
            )

            if not historical_logs.exists():
                return None

            # Parse JSON input_features and extract temperatures
            temps = []
            for log in historical_logs:
                try:
                    features = log.get_input_features()
                    if "Temp" in features:
                        temps.append(float(features["Temp"]))
                except Exception:
                    continue

            if not temps:
                return None

            # Calculate statistics
            avg_temp = sum(temps) / len(temps)
            max_temp = max(temps)
            min_temp = min(temps)
            current_temp = current_weather.get("current_temp", 20)

            comparison = {
                "current": current_temp,
                "historical_average": round(avg_temp, 1),
                "historical_max": round(max_temp, 1),
                "historical_min": round(min_temp, 1),
                "difference": round(current_temp - avg_temp, 1),
                "status": "typical",
                "sample_size": len(temps),
            }

            if abs(current_temp - avg_temp) > 5:
                if current_temp > avg_temp:
                    comparison["status"] = "warmer than usual"
                    comparison["note"] = (
                        f"It's {abs(current_temp - avg_temp):.1f}°C warmer than average for this time."
                    )
                else:
                    comparison["status"] = "cooler than usual"
                    comparison["note"] = (
                        f"It's {abs(current_temp - avg_temp):.1f}°C cooler than average for this time."
                    )
            else:
                comparison["note"] = "Temperature is typical for this time of year."

            return comparison
        except Exception as e:
            print(f"Error in historical comparison: {e}")
            import traceback

            traceback.print_exc()

        return None