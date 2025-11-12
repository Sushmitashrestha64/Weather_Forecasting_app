"""
Weather Prediction Service
===========================
This module handles loading ML models and making weather predictions
for temperature, humidity, and rain for the next 5 hours.
"""

import json
import os
import pickle
import sys
from datetime import datetime, timedelta

import joblib
import numpy as np
import pandas as pd
from django.conf import settings


# Custom unpickler to handle old module references
class RenameUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        # Redirect old ML_model references to sklearn
        if module == "Forecasting.ML_model":
            # Try to import from sklearn.ensemble first
            try:
                from sklearn import ensemble

                return getattr(ensemble, name)
            except (ImportError, AttributeError):
                pass

            # Try sklearn.linear_model
            try:
                from sklearn import linear_model

                return getattr(linear_model, name)
            except (ImportError, AttributeError):
                pass

            # Try sklearn.tree
            try:
                from sklearn import tree

                return getattr(tree, name)
            except (ImportError, AttributeError):
                pass

            # Try xgboost
            try:
                import xgboost

                return getattr(xgboost, name)
            except (ImportError, AttributeError):
                pass

        return super().find_class(module, name)


def renamed_load(file_obj):
    """Load pickled object with module name remapping"""
    return RenameUnpickler(file_obj).load()


class WeatherPredictor:
    """
    Service class for making weather predictions using trained ML models
    """

    def __init__(self):
        self.models_dir = os.path.join(settings.BASE_DIR, "Forecasting", "models")
        self.rain_model = None
        self.temp_model = None
        self.humidity_model = None
        self.feature_cols = None
        self._load_models()

    def _load_models(self):
        """Load all trained models from disk"""
        try:
            # Load rain prediction model
            rain_model_path = os.path.join(self.models_dir, "rain_model.joblib")
            if os.path.exists(rain_model_path):
                try:
                    with open(rain_model_path, "rb") as f:
                        self.rain_model = renamed_load(f)
                    # Validate model has required attributes
                    if not hasattr(self.rain_model, "predict"):
                        print("Warning: Rain model loaded but appears incomplete")
                        self.rain_model = None
                    else:
                        print("Rain model loaded successfully")
                except Exception as e:
                    print(f"Error loading rain model: {e}")
                    self.rain_model = None
            else:
                print(f"Rain model not found at {rain_model_path}")

            # Load temperature prediction model
            temp_model_path = os.path.join(self.models_dir, "temp_model.joblib")
            if os.path.exists(temp_model_path):
                try:
                    with open(temp_model_path, "rb") as f:
                        self.temp_model = renamed_load(f)
                    # Validate model has required attributes
                    if not hasattr(self.temp_model, "predict"):
                        print(
                            "Warning: Temperature model loaded but appears incomplete"
                        )
                        self.temp_model = None
                    else:
                        print("Temperature model loaded successfully")
                except Exception as e:
                    print(f"Error loading temperature model: {e}")
                    self.temp_model = None
            else:
                print(f"Temperature model not found at {temp_model_path}")

            # Load humidity prediction model
            humidity_model_path = os.path.join(self.models_dir, "humidity_model.joblib")
            if os.path.exists(humidity_model_path):
                try:
                    with open(humidity_model_path, "rb") as f:
                        self.humidity_model = renamed_load(f)
                    # Validate model has required attributes
                    if not hasattr(self.humidity_model, "predict"):
                        print("Warning: Humidity model loaded but appears incomplete")
                        self.humidity_model = None
                    else:
                        print("Humidity model loaded successfully")
                except Exception as e:
                    print(f"Error loading humidity model: {e}")
                    self.humidity_model = None
            else:
                print(f"Humidity model not found at {humidity_model_path}")

            # Load feature columns
            features_path = os.path.join(self.models_dir, "features.joblib")
            if os.path.exists(features_path):
                try:
                    self.feature_cols = joblib.load(features_path)
                    print(f"Feature columns loaded: {self.feature_cols}")
                except Exception as e:
                    print(f"Error loading features: {e}")
                    # Default features if loading fails
                    self.feature_cols = [
                        "Temp",
                        "MinTemp",
                        "MaxTemp",
                        "Humidity",
                        "Pressure",
                        "WindSpeed",
                        "WindDir",
                        "Rainfall",
                    ]
                    print(f"Using default feature columns: {self.feature_cols}")
            else:
                # Default features if file not found
                self.feature_cols = [
                    "Temp",
                    "MinTemp",
                    "MaxTemp",
                    "Humidity",
                    "Pressure",
                    "WindSpeed",
                    "WindDir",
                    "Rainfall",
                ]
                print(f"Using default feature columns: {self.feature_cols}")

        except Exception as e:
            print(f"Error loading models: {e}")
            # Don't raise - allow the service to continue with limited functionality
            print("Prediction service will use fallback/heuristic methods")

    def prepare_features(self, weather_data, model_type="rain"):
        """
        Prepare feature DataFrame from weather data for a specific model

        Args:
            weather_data: Dictionary containing current weather information
            model_type: Type of model ('rain', 'temp', or 'humidity')

        Returns:
            pandas DataFrame with required features
        """
        from datetime import datetime

        now = datetime.now()

        features = {
            "Temp": weather_data.get("current_temp", 25),
            "MinTemp": weather_data.get("temp_min", 20),
            "MaxTemp": weather_data.get("temp_max", 30),
            "Humidity": weather_data.get("humidity", 50),
            "Pressure": weather_data.get("pressure", 1013),
            "WindSpeed": weather_data.get("wind_speed", 0) * 3.6,  # Convert m/s to km/h
            "WindDir": weather_data.get("wind_dir", 0),
            "Rainfall": weather_data.get("rainfall", 0),
            "day_of_year": now.timetuple().tm_yday,
            "month": now.month,
            "day_of_week": now.weekday(),
        }

        # Create DataFrame
        df = pd.DataFrame([features])

        # Determine which feature set to use based on model type
        feature_list = self.feature_cols
        if isinstance(feature_list, dict):
            feature_list = feature_list.get(model_type, list(features.keys()))

        # Ensure all required feature columns are present
        for col in feature_list:
            if col not in df.columns:
                df[col] = 0

        # Select only the required features in the correct order
        try:
            df = df[feature_list]
        except KeyError as e:
            print(f"Warning: Feature mismatch for {model_type} model: {e}")
            # If some features are missing, use available ones
            available_features = [col for col in feature_list if col in df.columns]
            df = df[available_features]

        return df

    def predict_rain(self, weather_data):
        """
        Predict rain for the next period

        Args:
            weather_data: Dictionary containing current weather information

        Returns:
            dict with rain_prediction (0 or 1) and rain_probability (0-100)
        """
        if self.rain_model is None:
            # Use heuristic fallback
            humidity = weather_data.get("humidity", 50)
            clouds = weather_data.get("clouds", 0)
            pressure = weather_data.get("pressure", 1013)

            # Simple heuristic: high humidity + high clouds + low pressure = rain likely
            probability = 0.0
            if humidity > 80 and clouds > 70 and pressure < 1010:
                probability = 75.0
                prediction = 1
            elif humidity > 70 and clouds > 50:
                probability = 50.0
                prediction = 0
            elif humidity > 60:
                probability = 30.0
                prediction = 0
            else:
                probability = 10.0
                prediction = 0

            return {
                "rain_prediction": prediction,
                "rain_probability": round(probability, 2),
                "note": "Using heuristic (model not available)",
            }

        try:
            features_df = self.prepare_features(weather_data, model_type="rain")

            # Try multiple prediction approaches for compatibility
            prediction = None
            probability = 0.0

            # Approach 1: Try with DataFrame directly
            try:
                prediction = int(self.rain_model.predict(features_df)[0])
                if hasattr(self.rain_model, "predict_proba"):
                    proba = self.rain_model.predict_proba(features_df)[0]
                    probability = float(proba[1] * 100)
                else:
                    probability = float(prediction * 100)
            except Exception:
                # Approach 2: Try with numpy array
                try:
                    prediction = int(self.rain_model.predict(features_df.values)[0])
                    if hasattr(self.rain_model, "predict_proba"):
                        proba = self.rain_model.predict_proba(features_df.values)[0]
                        probability = float(proba[1] * 100)
                    else:
                        probability = float(prediction * 100)
                except Exception:
                    # Approach 3: Try with reshaped array
                    try:
                        features_array = features_df.values.reshape(1, -1)
                        prediction = int(self.rain_model.predict(features_array)[0])
                        if hasattr(self.rain_model, "predict_proba"):
                            proba = self.rain_model.predict_proba(features_array)[0]
                            probability = float(proba[1] * 100)
                        else:
                            probability = float(prediction * 100)
                    except Exception:
                        raise  # Re-raise to be caught by outer exception

            if prediction is not None:
                return {
                    "rain_prediction": int(prediction),
                    "rain_probability": round(probability, 2),
                }

        except Exception as e:
            # Silently use fallback (don't print traceback to reduce noise)
            pass

        # Fallback to heuristic
        humidity = weather_data.get("humidity", 50)
        clouds = weather_data.get("clouds", 0)
        pressure = weather_data.get("pressure", 1013)

        # Enhanced heuristic based on multiple factors
        probability = 0.0
        if humidity > 80 and clouds > 70 and pressure < 1010:
            probability = min(75.0 + (humidity - 80) * 0.5, 90.0)
        elif humidity > 70 and clouds > 50:
            probability = min(50.0 + (humidity - 70) * 0.8, 70.0)
        elif humidity > 60:
            probability = min(30.0 + (humidity - 60) * 0.5, 50.0)
        else:
            probability = max(10.0, humidity * 0.3)

        result = {
            "rain_prediction": 1 if probability > 60 else 0,
            "rain_probability": round(probability, 1),
            "method": "heuristic",
        }
        print(
            f"Rain Prediction: {'Rain Expected' if result['rain_prediction'] == 1 else 'No Rain'} ({result['rain_probability']}% probability)"
        )
        return result

    def predict_temperature_5_hours(self, weather_data):
        """
        Predict temperature for the next 5 hours

        Args:
            weather_data: Dictionary containing current weather information

        Returns:
            list of 5 temperature predictions with timestamps
        """
        current_temp = weather_data.get("current_temp", 25)

        if self.temp_model is None:
            # Use simple trend-based fallback
            predictions = []
            # Assume temperature changes slowly - ±0.5°C per hour
            for hour in range(1, 6):
                # Small random variation
                temp_change = (
                    (hour * 0.3) if datetime.now().hour < 14 else (hour * -0.3)
                )
                predicted_temp = current_temp + temp_change

                prediction_time = datetime.now() + timedelta(hours=hour)
                predictions.append(
                    {
                        "hour": hour,
                        "time": prediction_time.strftime("%I:%M %p"),
                        "temperature": round(float(predicted_temp), 1),
                        "timestamp": prediction_time.isoformat(),
                    }
                )
            return predictions

        try:
            predictions = []
            current_features_df = self.prepare_features(weather_data, model_type="temp")
            temp_features = (
                self.feature_cols.get("temp", [])
                if isinstance(self.feature_cols, dict)
                else []
            )

            for hour in range(1, 6):
                # Try multiple prediction approaches
                predicted_temp = None

                try:
                    # Approach 1: DataFrame
                    predicted_temp = float(
                        self.temp_model.predict(current_features_df)[0]
                    )
                except Exception:
                    try:
                        # Approach 2: NumPy array
                        predicted_temp = float(
                            self.temp_model.predict(current_features_df.values)[0]
                        )
                    except Exception:
                        try:
                            # Approach 3: Reshaped array
                            features_array = current_features_df.values.reshape(1, -1)
                            predicted_temp = float(
                                self.temp_model.predict(features_array)[0]
                            )
                        except Exception:
                            raise  # Re-raise to trigger fallback

                if predicted_temp is None:
                    raise Exception("All prediction approaches failed")

                # Create prediction entry with timestamp
                prediction_time = datetime.now() + timedelta(hours=hour)
                predictions.append(
                    {
                        "hour": hour,
                        "time": prediction_time.strftime("%I:%M %p"),
                        "temperature": round(predicted_temp, 1),
                        "timestamp": prediction_time.isoformat(),
                    }
                )

                # Update features for next iteration
                for i, col in enumerate(temp_features):
                    if col == "Temp":
                        current_features_df.iloc[0, i] = predicted_temp
                    elif col == "MinTemp":
                        current_features_df.iloc[0, i] = min(
                            float(current_features_df.iloc[0, i]), predicted_temp
                        )
                    elif col == "MaxTemp":
                        current_features_df.iloc[0, i] = max(
                            float(current_features_df.iloc[0, i]), predicted_temp
                        )

            return predictions

        except Exception as e:
            # Silently use fallback
            pass

        # Enhanced fallback with time-of-day awareness
        predictions = []
        current_hour = datetime.now().hour

        for hour in range(1, 6):
            future_hour = (current_hour + hour) % 24

            # Temperature typically rises until 2-3 PM, then falls
            if 6 <= future_hour <= 14:
                temp_change = hour * 0.4  # Rising
            elif 14 < future_hour <= 18:
                temp_change = (6 - hour) * 0.2  # Slight rise or stable
            else:
                temp_change = hour * -0.3  # Falling

            predicted_temp = current_temp + temp_change
            prediction_time = datetime.now() + timedelta(hours=hour)
            predictions.append(
                {
                    "hour": hour,
                    "time": prediction_time.strftime("%I:%M %p"),
                    "temperature": round(float(predicted_temp), 1),
                    "timestamp": prediction_time.isoformat(),
                    "method": "heuristic",
                }
            )

        # Log predictions
        print(f"Temperature 5-Hour Forecast:")
        for pred in predictions:
            print(f"  +{pred['hour']}hr ({pred['time']}): {pred['temperature']}°C")
        return predictions

    def predict_humidity_5_hours(self, weather_data):
        """
        Predict humidity for the next 5 hours

        Args:
            weather_data: Dictionary containing current weather information

        Returns:
            list of 5 humidity predictions with timestamps
        """
        current_humidity = weather_data.get("humidity", 50)

        if self.humidity_model is None:
            # Use simple trend-based fallback
            predictions = []
            # Assume humidity changes slowly - ±2% per hour
            for hour in range(1, 6):
                humidity_change = hour * 1.5 if current_humidity < 70 else hour * -1.5
                predicted_humidity = max(
                    0, min(100, current_humidity + humidity_change)
                )

                prediction_time = datetime.now() + timedelta(hours=hour)
                predictions.append(
                    {
                        "hour": hour,
                        "time": prediction_time.strftime("%I:%M %p"),
                        "humidity": round(float(predicted_humidity), 1),
                        "timestamp": prediction_time.isoformat(),
                    }
                )
            return predictions

        try:
            predictions = []
            current_features_df = self.prepare_features(
                weather_data, model_type="humidity"
            )
            humidity_features = (
                self.feature_cols.get("humidity", [])
                if isinstance(self.feature_cols, dict)
                else []
            )

            for hour in range(1, 6):
                # Try multiple prediction approaches
                predicted_humidity = None

                try:
                    # Approach 1: DataFrame
                    predicted_humidity = float(
                        self.humidity_model.predict(current_features_df)[0]
                    )
                except Exception:
                    try:
                        # Approach 2: NumPy array
                        predicted_humidity = float(
                            self.humidity_model.predict(current_features_df.values)[0]
                        )
                    except Exception:
                        try:
                            # Approach 3: Reshaped array
                            features_array = current_features_df.values.reshape(1, -1)
                            predicted_humidity = float(
                                self.humidity_model.predict(features_array)[0]
                            )
                        except Exception:
                            raise  # Re-raise to trigger fallback

                if predicted_humidity is None:
                    raise Exception("All prediction approaches failed")

                # Ensure humidity is within valid range (0-100)
                predicted_humidity = max(0, min(100, predicted_humidity))

                # Create prediction entry with timestamp
                prediction_time = datetime.now() + timedelta(hours=hour)
                predictions.append(
                    {
                        "hour": hour,
                        "time": prediction_time.strftime("%I:%M %p"),
                        "humidity": round(predicted_humidity, 1),
                        "timestamp": prediction_time.isoformat(),
                    }
                )

                # Update features for next iteration
                for i, col in enumerate(humidity_features):
                    if col == "Humidity":
                        current_features_df.iloc[0, i] = predicted_humidity

            return predictions

        except Exception as e:
            # Silently use fallback
            pass

        # Enhanced fallback with realistic humidity patterns
        predictions = []
        current_hour = datetime.now().hour

        for hour in range(1, 6):
            future_hour = (current_hour + hour) % 24

            # Humidity typically increases at night, decreases during day
            if 6 <= future_hour <= 14:
                humidity_change = hour * -1.5  # Decreasing
            elif 14 < future_hour <= 18:
                humidity_change = hour * -0.5  # Slight decrease
            else:
                humidity_change = hour * 1.0  # Increasing

            # Also factor in current humidity level
            if current_humidity > 80:
                humidity_change *= 0.5  # Less change when already high
            elif current_humidity < 40:
                humidity_change = abs(humidity_change)  # Tend to increase

            predicted_humidity = max(0, min(100, current_humidity + humidity_change))
            prediction_time = datetime.now() + timedelta(hours=hour)
            predictions.append(
                {
                    "hour": hour,
                    "time": prediction_time.strftime("%I:%M %p"),
                    "humidity": round(float(predicted_humidity), 1),
                    "timestamp": prediction_time.isoformat(),
                    "method": "heuristic",
                }
            )

        # Log predictions
        print(f"Humidity 5-Hour Forecast:")
        for pred in predictions:
            print(f"  +{pred['hour']}hr ({pred['time']}): {pred['humidity']}%")
        return predictions

    def get_comprehensive_predictions(self, weather_data):
        """
        Get all predictions (rain, temperature, humidity) for the next 5 hours

        Args:
            weather_data: Dictionary containing current weather information

        Returns:
            Dictionary containing all predictions
        """
        try:
            # Get rain prediction
            rain_pred = self.predict_rain(weather_data)

            # Get 5-hour temperature predictions
            temp_predictions = self.predict_temperature_5_hours(weather_data)

            # Get 5-hour humidity predictions
            humidity_predictions = self.predict_humidity_5_hours(weather_data)

            # Combine temperature and humidity predictions by hour
            hourly_predictions = []
            for i in range(5):
                if i < len(temp_predictions) and i < len(humidity_predictions):
                    prediction_entry = {
                        "hour": i + 1,
                        "time": temp_predictions[i]["time"],
                        "timestamp": temp_predictions[i]["timestamp"],
                        "temperature": temp_predictions[i]["temperature"],
                        "humidity": humidity_predictions[i]["humidity"],
                    }

                    # Add method indicator if using heuristic
                    if (
                        "method" in temp_predictions[i]
                        or "method" in humidity_predictions[i]
                    ):
                        prediction_entry["method"] = "heuristic"

                    hourly_predictions.append(prediction_entry)

            result = {
                "rain": rain_pred,
                "temperature_predictions": temp_predictions,
                "humidity_predictions": humidity_predictions,
                "hourly_predictions": hourly_predictions,
                "success": True,
            }

            # Log comprehensive predictions summary
            print(f"\n{'=' * 60}")
            print(f"5-Hour Weather Forecast Summary")
            print(f"{'=' * 60}")
            print(
                f"Rain: {'Expected' if rain_pred.get('rain_prediction') == 1 else 'Not Expected'} ({rain_pred.get('rain_probability', 0)}%)"
            )
            print(f"\nHourly Forecast:")
            for pred in hourly_predictions:
                print(
                    f"  +{pred['hour']}hr ({pred['time']}): {pred['temperature']}°C, {pred['humidity']}%"
                )
            print(f"{'=' * 60}\n")

            return result

        except Exception as e:
            print(f"Error getting comprehensive predictions: {e}")
            return {
                "rain": {"rain_prediction": 0, "rain_probability": 0.0},
                "temperature_predictions": [],
                "humidity_predictions": [],
                "hourly_predictions": [],
                "success": False,
                "error": str(e),
            }

    def save_predictions_to_log(self, city_name, weather_data, user=None):
        """
        Save predictions to database for tracking and analysis

        Args:
            city_name: Name of the city
            weather_data: Current weather data
            user: Django User object (optional)

        Returns:
            PredictionLog instance
        """
        from .models import PredictionLog

        try:
            # Get all predictions
            predictions = self.get_comprehensive_predictions(weather_data)

            # Prepare input features
            features = self.prepare_features(weather_data)
            input_features_dict = features.iloc[0].to_dict()

            # Extract prediction values
            rain_pred = predictions["rain"]["rain_prediction"]
            rain_prob = predictions["rain"]["rain_probability"]

            temp_preds = [
                p["temperature"] for p in predictions["temperature_predictions"]
            ]
            humidity_preds = [
                p["humidity"] for p in predictions["humidity_predictions"]
            ]

            # Create prediction log entry
            log = PredictionLog.objects.create(
                city_name=city_name,
                rain_model_version="v1.0",
                temp_model_version="v1.0",
                humidity_model_version="v1.0",
                input_features=json.dumps(input_features_dict),
                rain_prediction=rain_pred,
                rain_probability=rain_prob,
                temp_predictions=json.dumps(temp_preds),
                humidity_predictions=json.dumps(humidity_preds),
                user=user,
            )

            # Detailed logging
            print(f"\n{'─' * 60}")
            print(f"📝 Prediction Log Saved")
            print(f"{'─' * 60}")
            print(f"City: {city_name}")
            print(f"Log ID: {log.id}")
            print(f"User: {user.username if user else 'Anonymous'}")
            print(f"Timestamp: {log.prediction_date.strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"\nPredictions Stored:")
            print(f"  Rain: {'Yes' if rain_pred == 1 else 'No'} ({rain_prob}%)")
            print(f"  Temperature: {len(temp_preds)} hourly values")
            print(f"  Humidity: {len(humidity_preds)} hourly values")
            print(f"{'─' * 60}\n")

            return log

        except Exception as e:
            print(f"Error saving predictions to log: {e}")
            return None


# Singleton instance
_predictor_instance = None


def get_predictor():
    """
    Get or create singleton WeatherPredictor instance

    Returns:
        WeatherPredictor instance
    """
    global _predictor_instance
    if _predictor_instance is None:
        _predictor_instance = WeatherPredictor()
    return _predictor_instance
