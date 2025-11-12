# 5-Hour Weather Prediction Feature

## Overview

The Weather Forecasting App now includes an advanced **5-Hour Weather Prediction** feature that uses machine learning models to forecast temperature, humidity, and rain probability for the next 5 hours.

## Features

### 1. Rain Prediction
- **Binary Classification**: Predicts whether it will rain (Yes/No)
- **Probability Score**: Shows the likelihood of rain as a percentage (0-100%)
- **Visual Indicators**: Color-coded badges for quick understanding
  - 🌧️ Blue badge for "Rain Expected"
  - ☀️ Green badge for "No Rain Expected"

### 2. Hourly Temperature Forecast
- Predicts temperature for each of the next 5 hours
- Displays in Celsius with one decimal precision
- Shows the time for each prediction (e.g., "02:30 PM")
- Visual presentation with gradient-colored cards

### 3. Hourly Humidity Forecast
- Predicts humidity percentage for each of the next 5 hours
- Displayed alongside temperature in the same hourly card
- Values range from 0% to 100%

## How It Works

### Machine Learning Models

The prediction system uses three separate ML models:

1. **Rain Model** (`rain_model.joblib`)
   - Random Forest or XGBoost classifier
   - Predicts binary rain/no-rain outcome
   - Provides probability estimates

2. **Temperature Model** (`temp_model.joblib`)
   - Regression model for temperature prediction
   - Iteratively predicts next hour based on current conditions
   - Accounts for gradual temperature changes

3. **Humidity Model** (`humidity_model.joblib`)
   - Regression model for humidity prediction
   - Predicts humidity levels for each hour
   - Ensures values stay within valid range (0-100%)

### Input Features

The models use the following weather features:
- Current Temperature (°C)
- Minimum Temperature (°C)
- Maximum Temperature (°C)
- Humidity (%)
- Atmospheric Pressure (hPa)
- Wind Speed (km/h)
- Wind Direction (degrees)
- Rainfall (mm)

### Prediction Process

1. **Data Collection**: Current weather data is fetched from OpenWeather API
2. **Feature Preparation**: Weather data is transformed into model input format
3. **Model Inference**: Each model makes predictions independently
4. **Iterative Forecasting**: For temperature and humidity, each hour's prediction feeds into the next
5. **Result Aggregation**: All predictions are combined into a comprehensive forecast
6. **Database Logging**: Predictions are saved for accuracy tracking and analysis

## Technical Implementation

### Core Components

#### 1. `prediction_service.py`
The main prediction service module containing:

```python
class WeatherPredictor:
    - _load_models(): Loads trained ML models from disk
    - prepare_features(): Transforms weather data into model input
    - predict_rain(): Generates rain predictions
    - predict_temperature_5_hours(): Generates 5-hour temperature forecast
    - predict_humidity_5_hours(): Generates 5-hour humidity forecast
    - get_comprehensive_predictions(): Returns all predictions
    - save_predictions_to_log(): Saves predictions to database
```

#### 2. `advanced_views.py`
Updated to integrate predictions into the dashboard:
- Calls `get_predictor()` to get singleton predictor instance
- Generates predictions when city weather is requested
- Passes predictions to template context
- Handles errors gracefully with fallback behavior

#### 3. `weather_dashboard_simple.html`
Enhanced template displaying predictions:
- New "5-Hour Weather Forecast" section
- Rain prediction display with probability
- Hourly forecast cards for each of the 5 hours
- Responsive grid layout (5 columns on desktop, 3 on tablet, 2 on mobile)
- Beautiful gradient-styled prediction cards

### Database Schema

Predictions are logged in the `PredictionLog` model:

```python
class PredictionLog(models.Model):
    city_name = CharField()
    prediction_date = DateTimeField()
    rain_model_version = CharField()
    temp_model_version = CharField()
    humidity_model_version = CharField()
    input_features = TextField()  # JSON
    rain_prediction = IntegerField()
    rain_probability = FloatField()
    temp_predictions = TextField()  # JSON array of 5 values
    humidity_predictions = TextField()  # JSON array of 5 values
    user = ForeignKey(User, null=True)
```

## Usage

### For Users

1. **Access the Dashboard**: Navigate to the main weather dashboard
2. **Search for a City**: Enter a city name in the search box
3. **View Predictions**: Scroll down to the "5-Hour Weather Forecast" section
4. **Interpret Results**:
   - Check the rain prediction badge and probability
   - View each hourly forecast card for temperature and humidity
   - Use predictions to plan your next few hours

### For Developers

#### Getting Predictions Programmatically

```python
from Forecasting.prediction_service import get_predictor
from Forecasting.api_client import get_current_weather

# Get current weather
weather_data = get_current_weather("New York")

# Get predictor instance
predictor = get_predictor()

# Generate predictions
predictions = predictor.get_comprehensive_predictions(weather_data)

# Access specific predictions
rain_pred = predictions['rain']['rain_prediction']  # 0 or 1
rain_prob = predictions['rain']['rain_probability']  # 0-100
hourly = predictions['hourly_predictions']  # List of 5 hourly forecasts

# Save to database
predictor.save_predictions_to_log("New York", weather_data, user=request.user)
```

#### Adding New Features

To add new prediction features:

1. Train a new ML model with appropriate features
2. Save the model as a `.joblib` file in `Forecasting/models/`
3. Add loading logic in `WeatherPredictor._load_models()`
4. Create a prediction method (e.g., `predict_wind_5_hours()`)
5. Update `get_comprehensive_predictions()` to include new predictions
6. Update the template to display new predictions

## Model Training

### Requirements

- Historical weather data with hourly records
- Features: Temperature, Humidity, Pressure, Wind Speed, etc.
- Target variables: Rain (binary), Temperature (continuous), Humidity (continuous)

### Training Process

1. **Data Preparation**
   - Clean and preprocess historical weather data
   - Handle missing values
   - Feature engineering (if needed)

2. **Model Selection**
   - Rain: Random Forest Classifier or XGBoost
   - Temperature: Random Forest Regressor or LSTM
   - Humidity: Random Forest Regressor or LSTM

3. **Training & Validation**
   - Split data into train/test sets
   - Train models with cross-validation
   - Tune hyperparameters
   - Evaluate performance metrics

4. **Model Export**
   - Save trained models using `joblib`
   - Save feature column order
   - Document model versions and metrics

5. **Deployment**
   - Place `.joblib` files in `Forecasting/models/`
   - Update model version in code
   - Test predictions with sample data

## Performance Metrics

The system tracks prediction accuracy through:

- **Rain Prediction**: Accuracy, Precision, Recall, F1-Score
- **Temperature**: MAE (Mean Absolute Error), RMSE, R² Score
- **Humidity**: MAE, RMSE, R² Score

Metrics are stored in the `ModelMetadata` table and can be viewed in the Django admin panel.

## Limitations

1. **Model Accuracy**: Predictions are based on historical patterns and may not account for sudden weather changes
2. **Data Dependency**: Accuracy depends on the quality of input data from OpenWeather API
3. **Time Horizon**: 5-hour predictions; longer-term forecasts require different approaches
4. **Local Variations**: May not capture micro-climate effects
5. **Extreme Events**: May not accurately predict rare extreme weather events

## Future Enhancements

### Short Term
- [ ] Add confidence intervals for predictions
- [ ] Display prediction accuracy metrics to users
- [ ] Add wind speed and direction predictions
- [ ] Implement model versioning and A/B testing

### Medium Term
- [ ] Extend predictions to 12 hours
- [ ] Add daily forecast (7-day predictions)
- [ ] Implement ensemble methods for improved accuracy
- [ ] Add location-based model selection

### Long Term
- [ ] Real-time model retraining with new data
- [ ] Deep learning models (LSTM, GRU) for sequential predictions
- [ ] Integration with multiple weather APIs
- [ ] Custom models for specific regions

## Troubleshooting

### Predictions Not Showing

**Issue**: The 5-hour forecast section doesn't appear on the dashboard.

**Solutions**:
1. Check if `predictions_success` is True in the template context
2. Verify models exist in `Forecasting/models/` directory
3. Check Django logs for model loading errors
4. Ensure `hourly_predictions` list is not empty

### Incorrect Predictions

**Issue**: Predictions seem unrealistic or inaccurate.

**Solutions**:
1. Verify input data quality from OpenWeather API
2. Check feature preparation logic in `prepare_features()`
3. Review model training data and performance metrics
4. Consider retraining models with more recent data

### Model Loading Errors

**Issue**: Error message: "Model not loaded"

**Solutions**:
1. Ensure all `.joblib` files are present in `Forecasting/models/`
2. Check file permissions
3. Verify joblib package is installed
4. Check Django settings for correct `BASE_DIR`

## API Reference

### WeatherPredictor Methods

#### `get_predictor()`
Returns singleton instance of WeatherPredictor.

**Returns**: `WeatherPredictor` instance

#### `predict_rain(weather_data)`
Predicts rain for the next period.

**Parameters**:
- `weather_data` (dict): Current weather information

**Returns**: dict with `rain_prediction`, `rain_probability`

#### `predict_temperature_5_hours(weather_data)`
Predicts temperature for next 5 hours.

**Parameters**:
- `weather_data` (dict): Current weather information

**Returns**: list of 5 prediction dictionaries with `hour`, `time`, `temperature`, `timestamp`

#### `predict_humidity_5_hours(weather_data)`
Predicts humidity for next 5 hours.

**Parameters**:
- `weather_data` (dict): Current weather information

**Returns**: list of 5 prediction dictionaries with `hour`, `time`, `humidity`, `timestamp`

#### `get_comprehensive_predictions(weather_data)`
Gets all predictions at once.

**Parameters**:
- `weather_data` (dict): Current weather information

**Returns**: dict containing `rain`, `temperature_predictions`, `humidity_predictions`, `hourly_predictions`, `success`

## Contributing

To contribute to the prediction feature:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Update documentation
6. Submit a pull request

## License

This feature is part of the Weather Forecasting App and follows the same license as the main project.

## Support

For issues, questions, or suggestions:
- Open an issue on GitHub
- Contact the development team
- Check the main README for general support information

---

**Last Updated**: 2024
**Version**: 1.0.0
**Author**: Weather Forecasting App Team