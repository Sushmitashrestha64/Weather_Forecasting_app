import os
import joblib
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error, mean_absolute_error
from .data_preprocessing import outlier_check, preprocess_data, features
from django.conf import settings
from .ML_model import RandomForestClassifier, RandomForestRegressor

def prepare_data(df):
    TARGET_RAIN = 'RainToday'
    TARGET_TEMP = 'Temp'
    TARGET_HUMIDITY = 'Humidity'
    FEATURES_FOR_RAIN = [
        'MinTemp', 'MaxTemp', 'WindDir', 'Pressure',
        'WindSpeed', 'Temp', 'Humidity', 'day_of_year', 'month', 'day_of_week'
    ]
    FEATURES_FOR_TEMP = [
        'MinTemp', 'MaxTemp', 'Temp', 'Rainfall', 'WindDir', 'Pressure', 'WindSpeed', 'Humidity', 'day_of_year', 'month', 'day_of_week'
    ]  
    FEATURES_FOR_HUMIDITY = [
        'MinTemp', 'MaxTemp', 'Humidity', 'Rainfall', 'WindDir', 'Pressure', 'WindSpeed', 'Temp', 'day_of_year', 'month', 'day_of_week'
    ]
    
    X_rain = df[FEATURES_FOR_RAIN].values
    y_rain = df[TARGET_RAIN].values
    X_temp = df[FEATURES_FOR_TEMP].values
    y_temp = df[TARGET_TEMP].values
    X_humidity = df[FEATURES_FOR_HUMIDITY].values
    y_humidity = df[TARGET_HUMIDITY].values
    
    return X_rain, y_rain, FEATURES_FOR_RAIN, X_temp, y_temp, FEATURES_FOR_TEMP, X_humidity, y_humidity, FEATURES_FOR_HUMIDITY

def train_and_save_models(df):
    APP_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_PATH = os.path.join(APP_DIR, 'weather.csv')
    MODELS_DIR = os.path.join(APP_DIR, 'models')
    os.makedirs(MODELS_DIR, exist_ok=True)
    
    model_files = ['rain_model.joblib', 'temp_model.joblib', 'humidity_model.joblib', 'features.joblib']
    
    print(f"Attempting to read data from: {DATA_PATH}")
    print(f"Attempting to save models to: {MODELS_DIR}")
    
    if all(os.path.exists(os.path.join(MODELS_DIR, f)) for f in model_files):
        print("Loading existing models.")
        try:
            rain_model = joblib.load(os.path.join(MODELS_DIR, 'rain_model.joblib'))
            temp_model = joblib.load(os.path.join(MODELS_DIR, 'temp_model.joblib'))
            humidity_model = joblib.load(os.path.join(MODELS_DIR, 'humidity_model.joblib'))
            loaded_features = joblib.load(os.path.join(MODELS_DIR, 'features.joblib'))
            df = pd.read_csv(DATA_PATH)
            df_no_outliers = outlier_check(df)
            processed_df = preprocess_data(df_no_outliers)
            featured_df = features(processed_df)
            X_rain, y_rain, _, X_temp, y_temp, _, X_humidity, y_humidity, _ = prepare_data(featured_df)

            train_size = int(len(X_rain) * 0.8)
            X_test_rain = X_rain[train_size:]
            y_test_rain = y_rain[train_size:]
            X_test_temp = X_temp[train_size:]
            y_test_temp = y_temp[train_size:]
            X_test_humidity = X_humidity[train_size:]
            y_test_humidity = y_humidity[train_size:]
              
            results = {
                "rain_model": rain_model,
                "temp_model": temp_model,
                "humidity_model": humidity_model,
                "X_test_rain": X_test_rain,
                "y_test_rain": y_test_rain,
                "X_test_temp": X_test_temp,
                "y_test_temp": y_test_temp,
                "X_test_humidity": X_test_humidity,
                "y_test_humidity": y_test_humidity,
                "features": loaded_features
            }
            return "loaded", results
        except Exception as e:
            print(f"Error loading models: {e}")
            raise
    df = pd.read_csv(DATA_PATH)
    df_no_outliers = outlier_check(df)
    processed_df = preprocess_data(df_no_outliers)
    featured_df = features(processed_df)
    X_rain, y_rain, FEATURES_FOR_RAIN, X_temp, y_temp, FEATURES_FOR_TEMP, X_humidity, y_humidity, FEATURES_FOR_HUMIDITY= prepare_data(featured_df)
    FEATURES_DICT = {
        'rain': FEATURES_FOR_RAIN,
        'temp': FEATURES_FOR_TEMP,
        'humidity': FEATURES_FOR_HUMIDITY
    }
    train_size = int(len(X_rain) * 0.8)
    X_train_rain, X_test_rain = X_rain[:train_size], X_rain[train_size:]
    y_train_rain, y_test_rain = y_rain[:train_size], y_rain[train_size:]
    X_train_temp, X_test_temp = X_temp[:train_size], X_temp[train_size:]
    y_train_temp, y_test_temp = y_temp[:train_size], y_temp[train_size:]  
    X_train_humidity, X_test_humidity = X_humidity[:train_size], X_humidity[train_size:]
    y_train_humidity, y_test_humidity = y_humidity[:train_size], y_humidity[train_size:]
    
    print("\nTraining Models")
    n_features_rain = X_train_rain.shape[1] 
    n_feats_sqrt_rain = int(np.sqrt(n_features_rain))
    rain_model = RandomForestClassifier(n_trees=50, max_depth=10, n_feats=n_feats_sqrt_rain, random_state=42)
    rain_model.fit(X_train_rain, y_train_rain)
    print("Rain model trained.")

    n_features_temp = X_train_temp.shape[1]
    n_feats_sqrt_temp = int(np.sqrt(n_features_temp))
    temp_model = RandomForestRegressor(n_trees=50, max_depth=10, n_feats=n_feats_sqrt_temp, random_state=42)
    temp_model.fit(X_train_temp, y_train_temp)
    print("Temperature model trained.")

    n_features_humidity = X_train_humidity.shape[1]
    n_feats_sqrt_humidity = int(np.sqrt(n_features_humidity))   
    humidity_model = RandomForestRegressor(n_trees=50, max_depth=10, n_feats=n_feats_sqrt_humidity, random_state=42)
    humidity_model.fit(X_train_humidity, y_train_humidity)
    print("Humidity model trained.")

    try:
        joblib.dump(rain_model, os.path.join(MODELS_DIR, 'rain_model.joblib'))
        joblib.dump(temp_model, os.path.join(MODELS_DIR, 'temp_model.joblib'))
        joblib.dump(humidity_model, os.path.join(MODELS_DIR, 'humidity_model.joblib'))
        joblib.dump(FEATURES_DICT, os.path.join(MODELS_DIR, 'features.joblib'))
        print("\nAll models saved successfully.")
    except Exception as e:
        print(f"Error saving models: {e}")
        raise

    results = {
        "rain_model": rain_model,
        "temp_model": temp_model,
        "humidity_model": humidity_model,
        "X_test_rain": X_test_rain,
        "y_test_rain": y_test_rain,
        "X_test_temp": X_test_temp,
        "y_test_temp": y_test_temp,
        "X_test_humidity": X_test_humidity,
        "y_test_humidity": y_test_humidity,
        "features": FEATURES_DICT
    }
    return "trained", results

def evaluate_models(results):
    print("Model Evaluation on Unseen Test Data")
    rain_model = results["rain_model"]
    temp_model = results["temp_model"]
    humidity_model = results["humidity_model"]
    X_test_rain = results["X_test_rain"]
    y_test_rain = results["y_test_rain"]
    X_test_temp = results["X_test_temp"]
    y_test_temp = results["y_test_temp"]
    X_test_humidity = results["X_test_humidity"]
    y_test_humidity = results["y_test_humidity"]

    print("\nRain Model Evaluation")
    rain_preds = rain_model.predict(X_test_rain)
    print(f"Accuracy: {accuracy_score(y_test_rain, rain_preds):.2%}")
    print("\nClassification Report:")
    print(classification_report(y_test_rain, rain_preds, target_names=['No Rain', 'Rain']))

    print("\nTemperature Model Evaluation")
    temp_preds = temp_model.predict(X_test_temp)
    temp_mse = mean_squared_error(y_test_temp, temp_preds)
    temp_rmse = np.sqrt(temp_mse)
    temp_mae = mean_absolute_error(y_test_temp, temp_preds)
    print(f"Mean Absolute Error (MAE):   {temp_mae:.2f} °C")
    print(f"Root Mean Squared Error (RMSE): {temp_rmse:.2f}")

    print("\nHumidity Model Evaluation")
    humidity_preds = humidity_model.predict(X_test_humidity)
    humidity_mse = mean_squared_error(y_test_humidity, humidity_preds)
    humidity_rmse = np.sqrt(humidity_mse)
    humidity_mae = mean_absolute_error(y_test_humidity, humidity_preds)
    print(f"Mean Absolute Error (MAE):   {humidity_mae:.2f} %")
    print(f"Root Mean Squared Error (RMSE): {humidity_rmse:.2f}")

if __name__ == '__main__':
    try:
        status, results_dict = train_and_save_models()
        print(f"Models {'loaded' if status == 'loaded' else 'trained'} successfully.")
        if status != "failed":
            evaluate_models(results_dict)
            print(f"Evaluation completed using {'loaded' if status == 'loaded' else 'trained'} models.")
    except Exception as e:
        print(f"Failed to process data, train, or load models: {str(e)}")
        raise