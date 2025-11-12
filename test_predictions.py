#!/usr/bin/env python
"""
Test script for the 5-hour weather prediction feature
Run this to verify that the prediction service is working correctly
"""

import os
import sys

import django

# Setup Django environment
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "WeatherProject.settings")
django.setup()

from Forecasting.api_client import get_current_weather
from Forecasting.prediction_service import get_predictor


def test_predictions():
    """Test the prediction service with a sample city"""

    print("=" * 70)
    print("Testing 5-Hour Weather Prediction Feature")
    print("=" * 70)
    print()

    # Test city
    test_city = "London"

    print(f"1. Testing with city: {test_city}")
    print("-" * 70)

    # Get current weather
    print(f"   Fetching current weather for {test_city}...")
    weather_data = get_current_weather(test_city)

    if not weather_data:
        print(f"   ❌ ERROR: Could not fetch weather data for {test_city}")
        return False

    print(f"   ✅ Successfully fetched weather data")
    print(f"   Current Temperature: {weather_data['current_temp']}°C")
    print(f"   Humidity: {weather_data['humidity']}%")
    print(f"   Pressure: {weather_data['pressure']} hPa")
    print()

    # Get predictor instance
    print("2. Loading prediction models...")
    print("-" * 70)

    try:
        predictor = get_predictor()
        print("   ✅ Predictor instance created successfully")
    except Exception as e:
        print(f"   ❌ ERROR: Failed to load predictor: {e}")
        return False

    # Check if models are loaded
    print(
        f"   Rain Model Loaded: {'✅ Yes' if predictor.rain_model is not None else '❌ No'}"
    )
    print(
        f"   Temp Model Loaded: {'✅ Yes' if predictor.temp_model is not None else '❌ No'}"
    )
    print(
        f"   Humidity Model Loaded: {'✅ Yes' if predictor.humidity_model is not None else '❌ No'}"
    )
    print(f"   Features: {predictor.feature_cols}")
    print()

    # Test rain prediction
    print("3. Testing Rain Prediction...")
    print("-" * 70)

    try:
        rain_pred = predictor.predict_rain(weather_data)
        print(
            f"   ✅ Rain Prediction: {'Rain Expected' if rain_pred['rain_prediction'] == 1 else 'No Rain'}"
        )
        print(f"   Rain Probability: {rain_pred['rain_probability']}%")
    except Exception as e:
        print(f"   ❌ ERROR: {e}")
        rain_pred = None
    print()

    # Test temperature predictions
    print("4. Testing Temperature 5-Hour Forecast...")
    print("-" * 70)

    try:
        temp_predictions = predictor.predict_temperature_5_hours(weather_data)
        if temp_predictions:
            print(f"   ✅ Generated {len(temp_predictions)} hourly predictions:")
            for pred in temp_predictions:
                print(
                    f"      Hour +{pred['hour']}: {pred['temperature']}°C at {pred['time']}"
                )
        else:
            print("   ⚠️  No temperature predictions generated")
    except Exception as e:
        print(f"   ❌ ERROR: {e}")
        temp_predictions = []
    print()

    # Test humidity predictions
    print("5. Testing Humidity 5-Hour Forecast...")
    print("-" * 70)

    try:
        humidity_predictions = predictor.predict_humidity_5_hours(weather_data)
        if humidity_predictions:
            print(f"   ✅ Generated {len(humidity_predictions)} hourly predictions:")
            for pred in humidity_predictions:
                print(
                    f"      Hour +{pred['hour']}: {pred['humidity']}% at {pred['time']}"
                )
        else:
            print("   ⚠️  No humidity predictions generated")
    except Exception as e:
        print(f"   ❌ ERROR: {e}")
        humidity_predictions = []
    print()

    # Test comprehensive predictions
    print("6. Testing Comprehensive Predictions...")
    print("-" * 70)

    try:
        predictions = predictor.get_comprehensive_predictions(weather_data)

        print(f"   Success Status: {'✅ Yes' if predictions['success'] else '❌ No'}")
        print(f"   Rain Data: {predictions['rain']}")
        print(
            f"   Temperature Predictions: {len(predictions['temperature_predictions'])} entries"
        )
        print(
            f"   Humidity Predictions: {len(predictions['humidity_predictions'])} entries"
        )
        print(
            f"   Hourly Predictions: {len(predictions['hourly_predictions'])} entries"
        )

        if predictions["hourly_predictions"]:
            print()
            print("   Combined Hourly Forecast:")
            for pred in predictions["hourly_predictions"]:
                print(
                    f"      +{pred['hour']}hr ({pred['time']}): {pred['temperature']}°C, {pred['humidity']}%"
                )

    except Exception as e:
        print(f"   ❌ ERROR: {e}")
        import traceback

        traceback.print_exc()
        predictions = None
    print()

    # Test database logging
    print("7. Testing Database Logging...")
    print("-" * 70)

    try:
        log = predictor.save_predictions_to_log(test_city, weather_data, user=None)
        if log:
            print(f"   ✅ Predictions saved to database")
            print(f"   Log ID: {log.id}")
            print(f"   City: {log.city_name}")
            print(f"   Rain Prediction: {log.rain_prediction}")
            print(f"   Rain Probability: {log.rain_probability}%")
        else:
            print("   ⚠️  Failed to save predictions to database")
    except Exception as e:
        print(f"   ❌ ERROR: {e}")
        import traceback

        traceback.print_exc()
    print()

    # Summary
    print("=" * 70)
    print("Test Summary")
    print("=" * 70)

    success_count = 0
    total_tests = 7

    if weather_data:
        success_count += 1
    if (
        predictor.rain_model is not None
        and predictor.temp_model is not None
        and predictor.humidity_model is not None
    ):
        success_count += 1
    if rain_pred:
        success_count += 1
    if temp_predictions:
        success_count += 1
    if humidity_predictions:
        success_count += 1
    if predictions and predictions["success"]:
        success_count += 1
    if log:
        success_count += 1

    print(f"Tests Passed: {success_count}/{total_tests}")
    print()

    if success_count == total_tests:
        print(
            "🎉 All tests passed! The 5-hour prediction feature is working correctly."
        )
    elif success_count >= 5:
        print(
            "⚠️  Most tests passed. The feature is mostly working but may have minor issues."
        )
    else:
        print("❌ Several tests failed. Please check the error messages above.")

    print()
    return success_count == total_tests


if __name__ == "__main__":
    try:
        success = test_predictions()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n❌ Fatal Error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
