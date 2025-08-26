import pandas as pd
import numpy as np

def preprocess_data(df):
    df = df.copy()
    df['Date'] = pd.to_datetime(df['Date'], format='mixed')
    df.set_index('Date', inplace=True)
   
    df.dropna(axis=1, inplace=True)  
    if df['RainToday'].dtype == 'object':
        df['RainToday'] = df['RainToday'].map({'Yes': 1, 'No': 0})
    df['RainToday'] = df['RainToday'].fillna(0)
    df['RainToday'] = df['RainToday'].astype(int)
    numeric_cols = df.select_dtypes(include=np.number).columns
    df[numeric_cols] = df[numeric_cols].fillna(0)
    df.dropna(inplace=True)
    return df

def outlier_check(df):
    df_no_outliers = df.copy()
    outlier_check_cols = ['Temp', 'MinTemp', 'MaxTemp', 'WindSpeed', 'WindDir','Humidity', 'Pressure']
    missing_cols = [col for col in outlier_check_cols if col not in df_no_outliers.columns]
    if missing_cols:
        raise ValueError(f"Missing columns in DataFrame: {missing_cols}")
    initial_rows = len(df_no_outliers)
    for col in outlier_check_cols:
        Q1 = df_no_outliers[col].quantile(0.25)
        Q3 = df_no_outliers[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.0 * IQR
        upper_bound = Q3 + 1.0 * IQR
        df_no_outliers = df_no_outliers[(df_no_outliers[col] >= lower_bound) & (df_no_outliers[col] <= upper_bound)]
    return df_no_outliers


def features(df):
    featured_df = df.copy()
    if not pd.api.types.is_datetime64_any_dtype(featured_df.index):
        featured_df.index = pd.to_datetime(featured_df.index)
    if 'RainToday' in featured_df.columns and featured_df['RainToday'].dtype == 'object':
        featured_df['RainToday'] = featured_df['RainToday'].map({'Yes': 1, 'No': 0})

    featured_df['day_of_year'] = featured_df.index.dayofyear
    featured_df['month'] = featured_df.index.month
    featured_df['day_of_week'] = featured_df.index.dayofweek

    featured_df.dropna(inplace=True)
    return featured_df

