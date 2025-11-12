"""
Advanced Weather Views
======================
Views that utilize advanced analytics and algorithms
"""

import json
from datetime import datetime, timedelta

from django.contrib import messages
from django.contrib.auth.decorators import login_required
from django.http import JsonResponse
from django.shortcuts import redirect, render
from django.views.decorators.http import require_http_methods

from .api_client import get_current_weather
from .city_matcher import CityMatcher
from .models import PredictionLog, SearchHistory
from .news_fetcher import NewsFetcher
from .prediction_service import get_predictor
from .weather_analytics import WeatherAnalytics
from .weather_insights import WeatherInsights


@require_http_methods(["POST"])
def advanced_weather_analysis(request):
    """
    Advanced weather analysis with multiple algorithms
    """
    city = request.POST.get("city", "").strip()
    if not city:
        return JsonResponse({"error": "City name required"}, status=400)

    # Get current weather
    current_weather = get_current_weather(city)
    if not current_weather:
        return JsonResponse({"error": "City not found"}, status=404)

    analytics = WeatherAnalytics()

    # Calculate advanced metrics
    heat_index = analytics.calculate_heat_index(
        current_weather["current_temp"], current_weather["humidity"]
    )

    wind_chill = analytics.calculate_wind_chill(
        current_weather["current_temp"], current_weather["wind_speed"] * 3.6
    )

    dew_point = analytics.calculate_dew_point(
        current_weather["current_temp"], current_weather["humidity"]
    )

    comfort_index = analytics.calculate_comfort_index(
        current_weather["current_temp"],
        current_weather["humidity"],
        current_weather["wind_speed"] * 3.6,
    )

    # Advanced precipitation probability
    precip_prob = analytics.calculate_precipitation_probability_advanced(
        current_weather["humidity"],
        current_weather["pressure"],
        current_weather["current_temp"],
        current_weather["clouds"],
    )

    # Extreme weather detection
    extreme_warnings = analytics.detect_extreme_weather(current_weather)

    # Get recommendations
    weather_data_for_recommendations = {
        "current_temp": current_weather["current_temp"],
        "humidity": current_weather["humidity"],
        "wind_speed": current_weather["wind_speed"],
        "rain_prediction": 1 if precip_prob > 60 else 0,
        "description": current_weather["description"],
    }
    recommendations = analytics.get_weather_recommendation(
        weather_data_for_recommendations
    )

    # Get prediction accuracy if available
    accuracy = analytics.calculate_prediction_accuracy(current_weather["city"], days=7)

    response_data = {
        "city": current_weather["city"],
        "country": current_weather["country"],
        "basic_data": {
            "temperature": current_weather["current_temp"],
            "feels_like": current_weather["feels_like"],
            "humidity": current_weather["humidity"],
            "pressure": current_weather["pressure"],
            "wind_speed": current_weather["wind_speed"] * 3.6,
            "description": current_weather["description"],
        },
        "advanced_metrics": {
            "heat_index": heat_index,
            "wind_chill": wind_chill,
            "dew_point": dew_point,
            "comfort_index": comfort_index,
            "precipitation_probability": precip_prob,
        },
        "extreme_warnings": extreme_warnings,
        "recommendations": recommendations,
        "accuracy_metrics": accuracy,
    }

    return JsonResponse(response_data)


def weather_dashboard(request):
    """
    Enhanced weather dashboard with analytics
    """
    if request.method != "POST":
        return render(request, "weather_dashboard_simple.html")

    city = request.POST.get("city", "").strip()
    if not city:
        messages.error(request, "Please enter a city name")
        return render(request, "weather_dashboard_simple.html")

    try:
        # Try to correct city name if there are typos
        corrected_city, was_corrected, suggestions = CityMatcher.correct_city_name(
            city, auto_correct_threshold=0.85
        )

        # If auto-corrected, inform the user (only if actually different, not just case)
        if was_corrected and corrected_city.lower() != city.lower():
            messages.info(
                request,
                f"Showing results for '{corrected_city}' (corrected from '{city}')",
            )

        if was_corrected:
            city = corrected_city

        # Get current weather
        current_weather = get_current_weather(city)
        if not current_weather:
            # If weather fetch failed, show suggestions
            if suggestions:
                suggestion_cities = [s["city"] for s in suggestions[:3]]
                messages.error(
                    request,
                    f"City not found: {city}. Did you mean one of these cities?",
                )
                return render(
                    request,
                    "weather_dashboard_simple.html",
                    {"city_suggestions": suggestion_cities, "searched_city": city},
                )
            else:
                messages.error(
                    request,
                    f"City not found: {city}. Please check the spelling and try again.",
                )
            return render(request, "weather_dashboard_simple.html")

        analytics = WeatherAnalytics()
        insights_engine = WeatherInsights()
        news_fetcher = NewsFetcher()

        # Get weather predictor for 5-hour predictions
        try:
            predictor = get_predictor()
            predictions = predictor.get_comprehensive_predictions(current_weather)

            # Save predictions to log
            user = request.user if request.user.is_authenticated else None
            predictor.save_predictions_to_log(city, current_weather, user)
        except Exception as pred_error:
            print(f"Error getting predictions: {pred_error}")
            import traceback

            traceback.print_exc()
            predictions = {
                "rain": {"rain_prediction": 0, "rain_probability": 0.0},
                "hourly_predictions": [],
                "success": False,
            }

        # Extract prediction variables immediately to ensure they're available
        rain_prediction = predictions.get("rain", {}).get("rain_prediction", 0)
        rain_probability = predictions.get("rain", {}).get("rain_probability", 0.0)
        hourly_predictions = predictions.get("hourly_predictions", [])
        predictions_success = predictions.get("success", False)

        # Fetch global weather news (top 10)
        global_news = news_fetcher.get_trending_news(category="weather", max_results=10)

        # Fetch weather-specific news for location (top 10)
        location_news = news_fetcher.get_weather_news(city, max_results=10)

        # Calculate all advanced metrics
        heat_index = analytics.calculate_heat_index(
            current_weather["current_temp"], current_weather["humidity"]
        )

        wind_chill = analytics.calculate_wind_chill(
            current_weather["current_temp"], current_weather["wind_speed"] * 3.6
        )

        dew_point = analytics.calculate_dew_point(
            current_weather["current_temp"], current_weather["humidity"]
        )

        comfort_index = analytics.calculate_comfort_index(
            current_weather["current_temp"],
            current_weather["humidity"],
            current_weather["wind_speed"] * 3.6,
        )

        precip_prob = analytics.calculate_precipitation_probability_advanced(
            current_weather["humidity"],
            current_weather["pressure"],
            current_weather["current_temp"],
            current_weather["clouds"],
        )

        extreme_warnings = analytics.detect_extreme_weather(current_weather)

        weather_data_for_recommendations = {
            "current_temp": current_weather["current_temp"],
            "humidity": current_weather["humidity"],
            "wind_speed": current_weather["wind_speed"],
            "rain_prediction": 1 if precip_prob > 60 else 0,
            "description": current_weather["description"],
        }
        recommendations = analytics.get_weather_recommendation(
            weather_data_for_recommendations
        )

        # Generate intelligent insights
        forecast_data = None  # Can be populated with actual forecast if available
        intelligent_insights = insights_engine.get_comprehensive_insights(
            current_weather, forecast_data, city
        )

        # Get recent search history for trends
        recent_searches = SearchHistory.objects.filter(city_name__iexact=city).order_by(
            "-searched_at"
        )[:10]

        # Get prediction history
        prediction_history = PredictionLog.objects.filter(
            city_name__iexact=city
        ).order_by("-prediction_date")[:10]

        # Calculate trends if we have historical data
        trends = None
        volatility = None
        if prediction_history.count() >= 3:
            historical_data = []
            for pred in prediction_history:
                try:
                    input_features = json.loads(pred.input_features)
                    historical_data.append(
                        {
                            "temperature": input_features.get("Temp", 0),
                            "humidity": input_features.get("Humidity", 0),
                            "pressure": input_features.get("Pressure", 0),
                            "temp": input_features.get(
                                "Temp", 0
                            ),  # Add both for compatibility
                            "date": pred.prediction_date,
                        }
                    )
                except:
                    pass

            if len(historical_data) >= 3:
                try:
                    trends = analytics.analyze_weather_trends(historical_data)
                    volatility = analytics.calculate_weather_volatility(historical_data)
                except Exception as vol_error:
                    print(f"Error calculating volatility: {vol_error}")
                    volatility = None

        # Prepare context
        context = {
            "city": current_weather["city"],
            "country": current_weather["country"],
            "current_weather": current_weather,
            "heat_index": heat_index,
            "wind_chill": wind_chill,
            "dew_point": dew_point,
            "comfort_index": comfort_index,
            "precip_probability": precip_prob,
            "extreme_warnings": extreme_warnings,
            "recommendations": recommendations,
            "trends": trends,
            "volatility": volatility,
            "search_count": recent_searches.count(),
            "prediction_count": prediction_history.count(),
            # Add intelligent insights
            "clothing_insights": intelligent_insights.get("clothing"),
            "health_insights": intelligent_insights.get("health"),
            "activity_insights": intelligent_insights.get("activities"),
            "travel_insights": intelligent_insights.get("travel"),
            "energy_insights": intelligent_insights.get("energy"),
            "agriculture_insights": intelligent_insights.get("agriculture"),
            "photography_insights": intelligent_insights.get("photography"),
            "smart_alerts": intelligent_insights.get("alerts"),
            "historical_comparison": intelligent_insights.get("comparison"),
            # Add news
            "global_news": global_news,
            "location_news": location_news,
            # Add 5-hour predictions (already extracted above)
            "rain_prediction": rain_prediction,
            "rain_probability": rain_probability,
            "hourly_predictions": hourly_predictions,
            "predictions_success": predictions_success,
        }

        return render(request, "weather_dashboard_simple.html", context)

    except Exception as e:
        # Catch any errors and show user-friendly message
        print(f"Error in weather_dashboard view: {e}")
        import traceback

        traceback.print_exc()

        # Return partial data if we got this far
        if "current_weather" in locals() and current_weather:
            messages.warning(request, f"Showing weather data with limited analytics.")

            # Build minimal context with predictions if available
            minimal_context = {
                "city": current_weather.get("city", city),
                "country": current_weather.get("country", "Unknown"),
                "current_weather": current_weather,
                "heat_index": locals().get("heat_index", 0),
                "wind_chill": locals().get("wind_chill", 0),
                "dew_point": locals().get("dew_point", 0),
                "comfort_index": locals().get(
                    "comfort_index", {"score": 50, "level": "Moderate"}
                ),
                "precip_probability": locals().get("precip_prob", 0),
                "extreme_warnings": [],
                "recommendations": [],
                "trends": None,
                "volatility": None,
                "search_count": 0,
                "prediction_count": 0,
                "global_news": [],
                "location_news": [],
                # Include predictions - use the extracted variables
                "rain_prediction": locals().get("rain_prediction", 0),
                "rain_probability": locals().get("rain_probability", 0.0),
                "hourly_predictions": locals().get("hourly_predictions", []),
                "predictions_success": locals().get("predictions_success", False),
            }

            return render(request, "weather_dashboard_simple.html", minimal_context)
        else:
            messages.error(
                request,
                f"Unable to fetch weather data. Please try again.",
            )
            return render(request, "weather_dashboard_simple.html")


@login_required
def user_weather_analytics(request):
    """
    Personalized weather analytics for logged-in users
    """
    user = request.user

    # Get user's search history
    user_searches = SearchHistory.objects.filter(user=user).order_by("-searched_at")[
        :50
    ]

    # Get most searched cities
    top_cities = (
        SearchHistory.objects.filter(user=user)
        .values("city_name")
        .annotate(count=Count("id"))
        .order_by("-count")[:5]
    )

    # Get user's prediction logs
    user_predictions = PredictionLog.objects.filter(user=user).order_by(
        "-prediction_date"
    )[:20]

    # Calculate accuracy metrics
    analytics = WeatherAnalytics()
    accuracy_by_city = {}

    for city in top_cities:
        city_name = city["city_name"]
        accuracy = analytics.calculate_prediction_accuracy(city_name, days=30)
        if accuracy:
            accuracy_by_city[city_name] = accuracy

    context = {
        "total_searches": user_searches.count(),
        "top_cities": top_cities,
        "recent_searches": user_searches[:10],
        "total_predictions": user_predictions.count(),
        "accuracy_by_city": accuracy_by_city,
    }

    return render(request, "user_analytics.html", context)


@require_http_methods(["GET"])
def weather_comparison(request):
    """
    Compare weather across multiple cities
    """
    cities = request.GET.get("cities", "")
    if not cities:
        return JsonResponse(
            {"error": "Provide cities parameter (comma-separated)"}, status=400
        )

    city_list = [c.strip() for c in cities.split(",") if c.strip()]
    if len(city_list) < 2:
        return JsonResponse({"error": "Provide at least 2 cities"}, status=400)

    cities_data = []
    analytics = WeatherAnalytics()

    for city in city_list[:5]:  # Limit to 5 cities
        weather = get_current_weather(city)
        if weather:
            cities_data.append(
                {
                    "city": weather["city"],
                    "country": weather["country"],
                    "temp": weather["current_temp"],
                    "humidity": weather["humidity"],
                    "wind_speed": weather["wind_speed"] * 3.6,
                    "pressure": weather["pressure"],
                    "description": weather["description"],
                }
            )

    if len(cities_data) < 2:
        return JsonResponse({"error": "Could not find enough valid cities"}, status=404)

    comparison = analytics.compare_cities_weather(cities_data)

    response = {
        "cities": cities_data,
        "comparison": comparison,
    }

    return JsonResponse(response)


@require_http_methods(["POST"])
def detect_anomalies(request):
    """
    Detect weather anomalies for a given city
    """
    city = request.POST.get("city", "").strip()
    if not city:
        return JsonResponse({"error": "City name required"}, status=400)

    # Get current weather
    current_weather = get_current_weather(city)
    if not current_weather:
        return JsonResponse({"error": "City not found"}, status=404)

    # Get historical data from predictions
    historical_predictions = PredictionLog.objects.filter(
        city_name__iexact=city
    ).order_by("-prediction_date")[:30]

    if historical_predictions.count() < 5:
        return JsonResponse(
            {
                "message": "Not enough historical data for anomaly detection",
                "anomalies": [],
            }
        )

    # Calculate historical averages
    historical_data = []
    for pred in historical_predictions:
        try:
            features = json.loads(pred.input_features)
            historical_data.append(
                {
                    "temperature": features.get("Temp", 0),
                    "humidity": features.get("Humidity", 0),
                    "pressure": features.get("Pressure", 0),
                    "wind_speed": features.get("WindSpeed", 0),
                }
            )
        except:
            pass

    if not historical_data:
        return JsonResponse(
            {"message": "Could not parse historical data", "anomalies": []}
        )

    # Calculate statistics
    import numpy as np

    historical_avg = {
        "temperature": {
            "mean": np.mean([d["temperature"] for d in historical_data]),
            "std": np.std([d["temperature"] for d in historical_data]),
        },
        "humidity": {
            "mean": np.mean([d["humidity"] for d in historical_data]),
            "std": np.std([d["humidity"] for d in historical_data]),
        },
        "pressure": {
            "mean": np.mean([d["pressure"] for d in historical_data]),
            "std": np.std([d["pressure"] for d in historical_data]),
        },
        "wind_speed": {
            "mean": np.mean([d["wind_speed"] for d in historical_data]),
            "std": np.std([d["wind_speed"] for d in historical_data]),
        },
    }

    current_data = {
        "temperature": current_weather["current_temp"],
        "humidity": current_weather["humidity"],
        "pressure": current_weather["pressure"],
        "wind_speed": current_weather["wind_speed"] * 3.6,
    }

    analytics = WeatherAnalytics()
    anomalies = analytics.detect_weather_anomalies(current_data, historical_avg)

    return JsonResponse(
        {
            "city": current_weather["city"],
            "current_weather": current_data,
            "historical_averages": {
                k: {"mean": round(v["mean"], 2), "std": round(v["std"], 2)}
                for k, v in historical_avg.items()
            },
            "anomalies": anomalies,
        }
    )


@require_http_methods(["GET"])
def weather_trends_api(request):
    """
    Get weather trends for a city
    """
    city = request.GET.get("city", "")
    days = int(request.GET.get("days", 7))

    if not city:
        return JsonResponse({"error": "City parameter required"}, status=400)

    # Get historical predictions
    from_date = datetime.now() - timedelta(days=days)
    predictions = PredictionLog.objects.filter(
        city_name__iexact=city, prediction_date__gte=from_date
    ).order_by("-prediction_date")

    if predictions.count() < 2:
        return JsonResponse(
            {"message": "Not enough data for trend analysis", "trends": None}
        )

    historical_data = []
    for pred in predictions:
        try:
            features = json.loads(pred.input_features)
            historical_data.append(
                {
                    "temperature": features.get("Temp", 0),
                    "humidity": features.get("Humidity", 0),
                    "pressure": features.get("Pressure", 0),
                    "date": pred.prediction_date.isoformat(),
                }
            )
        except:
            pass

    if len(historical_data) < 2:
        return JsonResponse({"message": "Could not parse data", "trends": None})

    analytics = WeatherAnalytics()
    trends = analytics.analyze_weather_trends(historical_data)
    volatility = analytics.calculate_weather_volatility(historical_data)

    return JsonResponse(
        {
            "city": city,
            "period_days": days,
            "data_points": len(historical_data),
            "trends": trends,
            "volatility": volatility,
            "historical_data": historical_data,
        }
    )
