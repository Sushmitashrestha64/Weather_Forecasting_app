import json

from django.contrib.auth.models import User
from django.db import models
from django.utils import timezone


class UserProfile(models.Model):
    """Extended user profile for weather app preferences"""

    user = models.OneToOneField(User, on_delete=models.CASCADE, related_name="profile")
    preferred_unit = models.CharField(
        max_length=10,
        choices=[("celsius", "Celsius"), ("fahrenheit", "Fahrenheit")],
        default="celsius",
    )
    preferred_language = models.CharField(
        max_length=10, choices=[("en", "English"), ("ne", "Nepali")], default="en"
    )
    email_alerts = models.BooleanField(default=False)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return f"{self.user.username}'s Profile"

    class Meta:
        verbose_name = "User Profile"
        verbose_name_plural = "User Profiles"


class FavoriteCity(models.Model):
    """User's favorite cities for quick access"""

    user = models.ForeignKey(
        User, on_delete=models.CASCADE, related_name="favorite_cities"
    )
    city_name = models.CharField(max_length=100)
    country = models.CharField(max_length=100, blank=True)
    added_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.user.username} - {self.city_name}"

    class Meta:
        verbose_name = "Favorite City"
        verbose_name_plural = "Favorite Cities"
        unique_together = ["user", "city_name"]
        ordering = ["-added_at"]


class SearchHistory(models.Model):
    """Track user search history"""

    user = models.ForeignKey(
        User,
        on_delete=models.CASCADE,
        related_name="search_history",
        null=True,
        blank=True,
    )
    city_name = models.CharField(max_length=100)
    country = models.CharField(max_length=100, blank=True)
    searched_at = models.DateTimeField(auto_now_add=True)
    session_key = models.CharField(max_length=40, blank=True, null=True)

    def __str__(self):
        user_str = self.user.username if self.user else "Anonymous"
        return f"{user_str} searched {self.city_name} at {self.searched_at}"

    class Meta:
        verbose_name = "Search History"
        verbose_name_plural = "Search Histories"
        ordering = ["-searched_at"]


class ModelMetadata(models.Model):
    """Store metadata about trained ML models"""

    MODEL_TYPES = [
        ("rain_rf", "Rain Random Forest"),
        ("rain_xgb", "Rain XGBoost"),
        ("temp_rf", "Temperature Random Forest"),
        ("temp_lstm", "Temperature LSTM"),
        ("humidity_rf", "Humidity Random Forest"),
        ("humidity_lstm", "Humidity LSTM"),
    ]

    model_type = models.CharField(max_length=20, choices=MODEL_TYPES)
    version = models.CharField(max_length=20)
    file_path = models.CharField(max_length=255)

    # Performance metrics
    accuracy = models.FloatField(null=True, blank=True)
    mae = models.FloatField(null=True, blank=True, help_text="Mean Absolute Error")
    rmse = models.FloatField(null=True, blank=True, help_text="Root Mean Squared Error")
    r2_score = models.FloatField(null=True, blank=True, help_text="R² Score")
    f1_score = models.FloatField(null=True, blank=True)
    precision = models.FloatField(null=True, blank=True)
    recall = models.FloatField(null=True, blank=True)

    # Training details
    training_samples = models.IntegerField(null=True, blank=True)
    test_samples = models.IntegerField(null=True, blank=True)
    hyperparameters = models.TextField(help_text="JSON string of hyperparameters")
    features_used = models.TextField(help_text="JSON string of feature names")

    # Status
    is_active = models.BooleanField(default=False)
    trained_at = models.DateTimeField(auto_now_add=True)
    trained_by = models.ForeignKey(
        User, on_delete=models.SET_NULL, null=True, blank=True
    )

    notes = models.TextField(blank=True)

    def __str__(self):
        return f"{self.get_model_type_display()} - {self.version} ({'Active' if self.is_active else 'Inactive'})"

    def get_hyperparameters(self):
        """Return hyperparameters as dictionary"""
        try:
            return json.loads(self.hyperparameters)
        except:
            return {}

    def get_features(self):
        """Return features as list"""
        try:
            return json.loads(self.features_used)
        except:
            return []

    class Meta:
        verbose_name = "Model Metadata"
        verbose_name_plural = "Model Metadata"
        ordering = ["-trained_at"]
        unique_together = ["model_type", "version"]


class WeatherAlert(models.Model):
    """Store weather alerts triggered for users"""

    ALERT_TYPES = [
        ("high_temp", "High Temperature"),
        ("low_temp", "Low Temperature"),
        ("rain", "Rain Expected"),
        ("high_humidity", "High Humidity"),
        ("storm", "Storm Warning"),
    ]

    SEVERITY_LEVELS = [
        ("info", "Information"),
        ("warning", "Warning"),
        ("critical", "Critical"),
    ]

    user = models.ForeignKey(
        User,
        on_delete=models.CASCADE,
        related_name="weather_alerts",
        null=True,
        blank=True,
    )
    city_name = models.CharField(max_length=100)
    alert_type = models.CharField(max_length=20, choices=ALERT_TYPES)
    severity = models.CharField(max_length=10, choices=SEVERITY_LEVELS, default="info")
    message = models.TextField()
    threshold_value = models.FloatField(
        help_text="The threshold that triggered the alert"
    )
    actual_value = models.FloatField(
        help_text="The actual value that exceeded threshold"
    )

    created_at = models.DateTimeField(auto_now_add=True)
    is_read = models.BooleanField(default=False)
    is_sent = models.BooleanField(default=False)
    sent_at = models.DateTimeField(null=True, blank=True)

    def __str__(self):
        user_str = self.user.username if self.user else "Anonymous"
        return f"{user_str} - {self.get_alert_type_display()} for {self.city_name}"

    def mark_as_read(self):
        """Mark alert as read"""
        self.is_read = True
        self.save()

    def mark_as_sent(self):
        """Mark alert as sent"""
        self.is_sent = True
        self.sent_at = timezone.now()
        self.save()

    class Meta:
        verbose_name = "Weather Alert"
        verbose_name_plural = "Weather Alerts"
        ordering = ["-created_at"]


class PredictionLog(models.Model):
    """Log predictions for analysis and improvement"""

    city_name = models.CharField(max_length=100)
    prediction_date = models.DateTimeField(auto_now_add=True)

    # Models used
    rain_model_version = models.CharField(max_length=20)
    temp_model_version = models.CharField(max_length=20)
    humidity_model_version = models.CharField(max_length=20)

    # Input features (JSON)
    input_features = models.TextField(help_text="JSON string of input features")

    # Predictions
    rain_prediction = models.IntegerField(help_text="0 or 1")
    rain_probability = models.FloatField(null=True, blank=True)
    temp_predictions = models.TextField(
        help_text="JSON array of 5 temperature predictions"
    )
    humidity_predictions = models.TextField(
        help_text="JSON array of 5 humidity predictions"
    )

    # Confidence intervals (if available)
    temp_lower_bounds = models.TextField(null=True, blank=True)
    temp_upper_bounds = models.TextField(null=True, blank=True)
    humidity_lower_bounds = models.TextField(null=True, blank=True)
    humidity_upper_bounds = models.TextField(null=True, blank=True)

    # User context
    user = models.ForeignKey(User, on_delete=models.SET_NULL, null=True, blank=True)

    def __str__(self):
        return f"Prediction for {self.city_name} at {self.prediction_date}"

    def get_input_features(self):
        """Return input features as dictionary"""
        try:
            return json.loads(self.input_features)
        except:
            return {}

    def get_temp_predictions(self):
        """Return temperature predictions as list"""
        try:
            return json.loads(self.temp_predictions)
        except:
            return []

    def get_humidity_predictions(self):
        """Return humidity predictions as list"""
        try:
            return json.loads(self.humidity_predictions)
        except:
            return []

    class Meta:
        verbose_name = "Prediction Log"
        verbose_name_plural = "Prediction Logs"
        ordering = ["-prediction_date"]


class ModelComparisonResult(models.Model):
    """Store comparison results between different models"""

    comparison_date = models.DateTimeField(auto_now_add=True)
    model_type = models.CharField(
        max_length=50, help_text="e.g., 'Temperature', 'Humidity', 'Rain'"
    )

    # Model A
    model_a_name = models.CharField(max_length=100)
    model_a_version = models.CharField(max_length=20)
    model_a_mae = models.FloatField(null=True, blank=True)
    model_a_rmse = models.FloatField(null=True, blank=True)
    model_a_r2 = models.FloatField(null=True, blank=True)
    model_a_accuracy = models.FloatField(null=True, blank=True)

    # Model B
    model_b_name = models.CharField(max_length=100)
    model_b_version = models.CharField(max_length=20)
    model_b_mae = models.FloatField(null=True, blank=True)
    model_b_rmse = models.FloatField(null=True, blank=True)
    model_b_r2 = models.FloatField(null=True, blank=True)
    model_b_accuracy = models.FloatField(null=True, blank=True)

    # Winner
    winner = models.CharField(
        max_length=10, choices=[("A", "Model A"), ("B", "Model B"), ("TIE", "Tie")]
    )
    notes = models.TextField(blank=True)

    def __str__(self):
        return f"{self.model_type}: {self.model_a_name} vs {self.model_b_name} ({self.comparison_date.date()})"

    class Meta:
        verbose_name = "Model Comparison Result"
        verbose_name_plural = "Model Comparison Results"
        ordering = ["-comparison_date"]
