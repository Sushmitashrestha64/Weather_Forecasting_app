"""
ML Model Compatibility Module
==============================
This module exists for backward compatibility with pickled models
that were trained with references to Forecasting.ML_model.

It re-exports sklearn classes that may be referenced in old models.
"""

# Import sklearn classes that might be referenced in pickled models
from sklearn.ensemble import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import Lasso, LinearRegression, LogisticRegression, Ridge
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

# Try to import XGBoost if available
try:
    from xgboost import XGBClassifier, XGBRegressor
except ImportError:
    XGBClassifier = None
    XGBRegressor = None

# Try to import LSTM models if available
try:
    from keras.layers import LSTM, Dense, Dropout
    from keras.models import Sequential
except ImportError:
    Sequential = None
    LSTM = None
    Dense = None
    Dropout = None


# Dummy classes for backward compatibility with pickled models
class Node:
    """Compatibility class for custom decision tree nodes in old pickled models"""

    def __init__(self, *args, **kwargs):
        self.__dict__.update(kwargs)
        pass


class WeatherPredictor:
    """Compatibility class for old pickled models"""

    def __init__(self, *args, **kwargs):
        self.__dict__.update(kwargs)
        pass


class RainPredictor:
    """Compatibility class for old pickled models"""

    def __init__(self, *args, **kwargs):
        self.__dict__.update(kwargs)
        pass


class TemperaturePredictor:
    """Compatibility class for old pickled models"""

    def __init__(self, *args, **kwargs):
        self.__dict__.update(kwargs)
        pass


class HumidityPredictor:
    """Compatibility class for old pickled models"""

    def __init__(self, *args, **kwargs):
        self.__dict__.update(kwargs)
        pass


class ModelWrapper:
    """Compatibility class for old pickled models"""

    def __init__(self, *args, **kwargs):
        self.__dict__.update(kwargs)
        pass


class DecisionTree:
    """Compatibility class for custom decision trees in old pickled models"""

    def __init__(self, *args, **kwargs):
        self.__dict__.update(kwargs)
        pass


class TreeNode:
    """Compatibility class for tree nodes in old pickled models"""

    def __init__(self, *args, **kwargs):
        self.__dict__.update(kwargs)
        pass


class Leaf:
    """Compatibility class for leaf nodes in old pickled models"""

    def __init__(self, *args, **kwargs):
        self.__dict__.update(kwargs)
        pass
