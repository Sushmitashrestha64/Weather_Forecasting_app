# Forecasting/urls.py
from django.urls import path

from . import advanced_views

urlpatterns = [
    # Dashboard at root
    path("", advanced_views.weather_dashboard, name="weather_dashboard"),
    # Advanced analytics views
    path("analytics/", advanced_views.user_weather_analytics, name="user_analytics"),
    # API endpoints
    path(
        "api/advanced-analysis/",
        advanced_views.advanced_weather_analysis,
        name="advanced_analysis",
    ),
    path(
        "api/comparison/", advanced_views.weather_comparison, name="weather_comparison"
    ),
    path("api/anomalies/", advanced_views.detect_anomalies, name="detect_anomalies"),
    path("api/trends/", advanced_views.weather_trends_api, name="weather_trends"),
]
