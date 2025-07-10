import pytz
import pandas as pd
from datetime import timedelta

def predict_future(model, current_features, features_list, start_time):
    predictions = [current_features['Temp']] if 'Temp' in current_features else [current_features['Humidity']]
    feature_data = current_features.copy()
    
    for i in range(5):
        future_time = start_time + timedelta(hours=i+1)
        feature_data['day_of_year'] = future_time.timetuple().tm_yday
        feature_data['month'] = future_time.month
        feature_data['day_of_week'] = future_time.weekday()
        
        prediction_df = pd.DataFrame([feature_data])[features_list].fillna(0)
        next_value = model.predict(prediction_df.values)[0]
        predictions.append(next_value)
        if 'Temp' in current_features:
            feature_data['Temp'] = next_value
        else:
            feature_data['Humidity'] = next_value

    return predictions[1:]
