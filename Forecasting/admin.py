from django.contrib import admin
from django.db.models import Avg, Count
from django.utils.html import format_html

from .models import (
    FavoriteCity,
    ModelComparisonResult,
    ModelMetadata,
    PredictionLog,
    SearchHistory,
    UserProfile,
    WeatherAlert,
)


@admin.register(UserProfile)
class UserProfileAdmin(admin.ModelAdmin):
    list_display = [
        "user",
        "preferred_unit",
        "preferred_language",
        "email_alerts",
        "created_at",
    ]
    list_filter = ["preferred_unit", "preferred_language", "email_alerts", "created_at"]
    search_fields = ["user__username", "user__email"]
    readonly_fields = ["created_at", "updated_at"]

    fieldsets = (
        ("User Information", {"fields": ("user",)}),
        (
            "Preferences",
            {"fields": ("preferred_unit", "preferred_language", "email_alerts")},
        ),
        (
            "Timestamps",
            {"fields": ("created_at", "updated_at"), "classes": ("collapse",)},
        ),
    )


@admin.register(FavoriteCity)
class FavoriteCityAdmin(admin.ModelAdmin):
    list_display = ["user", "city_name", "country", "added_at"]
    list_filter = ["added_at", "country"]
    search_fields = ["user__username", "city_name", "country"]
    readonly_fields = ["added_at"]
    date_hierarchy = "added_at"


@admin.register(SearchHistory)
class SearchHistoryAdmin(admin.ModelAdmin):
    list_display = ["get_user_display", "city_name", "country", "searched_at"]
    list_filter = ["searched_at", "country"]
    search_fields = ["user__username", "city_name", "country", "session_key"]
    readonly_fields = ["searched_at"]
    date_hierarchy = "searched_at"

    def get_user_display(self, obj):
        if obj.user:
            return obj.user.username
        return "Anonymous"

    get_user_display.short_description = "User"

    def get_queryset(self, request):
        qs = super().get_queryset(request)
        return qs.select_related("user")


@admin.register(ModelMetadata)
class ModelMetadataAdmin(admin.ModelAdmin):
    list_display = [
        "model_type",
        "version",
        "status_badge",
        "performance_summary",
        "trained_at",
        "trained_by",
    ]
    list_filter = ["model_type", "is_active", "trained_at"]
    search_fields = ["model_type", "version", "notes"]
    readonly_fields = [
        "trained_at",
        "get_hyperparameters_display",
        "get_features_display",
    ]

    fieldsets = (
        (
            "Model Information",
            {"fields": ("model_type", "version", "file_path", "is_active")},
        ),
        (
            "Performance Metrics",
            {
                "fields": (
                    "accuracy",
                    "mae",
                    "rmse",
                    "r2_score",
                    "f1_score",
                    "precision",
                    "recall",
                )
            },
        ),
        (
            "Training Details",
            {
                "fields": (
                    "training_samples",
                    "test_samples",
                    "trained_by",
                    "trained_at",
                )
            },
        ),
        (
            "Configuration",
            {
                "fields": ("get_hyperparameters_display", "get_features_display"),
                "classes": ("collapse",),
            },
        ),
        ("Notes", {"fields": ("notes",), "classes": ("collapse",)}),
    )

    def status_badge(self, obj):
        if obj.is_active:
            return format_html(
                '<span style="background-color: #28a745; color: white; padding: 3px 10px; border-radius: 3px;">Active</span>'
            )
        return format_html(
            '<span style="background-color: #6c757d; color: white; padding: 3px 10px; border-radius: 3px;">Inactive</span>'
        )

    status_badge.short_description = "Status"

    def performance_summary(self, obj):
        metrics = []
        if obj.accuracy is not None:
            metrics.append(f"Acc: {obj.accuracy:.2%}")
        if obj.mae is not None:
            metrics.append(f"MAE: {obj.mae:.2f}")
        if obj.rmse is not None:
            metrics.append(f"RMSE: {obj.rmse:.2f}")
        if obj.r2_score is not None:
            metrics.append(f"R²: {obj.r2_score:.3f}")
        return " | ".join(metrics) if metrics else "N/A"

    performance_summary.short_description = "Performance"

    def get_hyperparameters_display(self, obj):
        import json

        try:
            params = json.loads(obj.hyperparameters)
            return format_html("<pre>{}</pre>", json.dumps(params, indent=2))
        except:
            return obj.hyperparameters

    get_hyperparameters_display.short_description = "Hyperparameters"

    def get_features_display(self, obj):
        import json

        try:
            features = json.loads(obj.features_used)
            return format_html("<pre>{}</pre>", "\n".join(features))
        except:
            return obj.features_used

    get_features_display.short_description = "Features Used"

    actions = ["activate_models", "deactivate_models"]

    def activate_models(self, request, queryset):
        # Deactivate all models of the same type first
        for obj in queryset:
            ModelMetadata.objects.filter(model_type=obj.model_type).update(
                is_active=False
            )
        # Activate selected models
        updated = queryset.update(is_active=True)
        self.message_user(request, f"{updated} model(s) activated successfully.")

    activate_models.short_description = "Activate selected models"

    def deactivate_models(self, request, queryset):
        updated = queryset.update(is_active=False)
        self.message_user(request, f"{updated} model(s) deactivated successfully.")

    deactivate_models.short_description = "Deactivate selected models"


@admin.register(WeatherAlert)
class WeatherAlertAdmin(admin.ModelAdmin):
    list_display = [
        "get_user_display",
        "city_name",
        "alert_type",
        "severity_badge",
        "created_at",
        "read_status",
        "sent_status",
    ]
    list_filter = ["alert_type", "severity", "is_read", "is_sent", "created_at"]
    search_fields = ["user__username", "city_name", "message"]
    readonly_fields = ["created_at", "sent_at"]
    date_hierarchy = "created_at"

    fieldsets = (
        (
            "Alert Information",
            {"fields": ("user", "city_name", "alert_type", "severity")},
        ),
        ("Details", {"fields": ("message", "threshold_value", "actual_value")}),
        ("Status", {"fields": ("is_read", "is_sent", "created_at", "sent_at")}),
    )

    def get_user_display(self, obj):
        if obj.user:
            return obj.user.username
        return "Anonymous"

    get_user_display.short_description = "User"

    def severity_badge(self, obj):
        colors = {
            "info": "#17a2b8",
            "warning": "#ffc107",
            "critical": "#dc3545",
        }
        color = colors.get(obj.severity, "#6c757d")
        return format_html(
            '<span style="background-color: {}; color: white; padding: 3px 10px; border-radius: 3px;">{}</span>',
            color,
            obj.get_severity_display(),
        )

    severity_badge.short_description = "Severity"

    def read_status(self, obj):
        if obj.is_read:
            return format_html('<span style="color: green;">✓ Read</span>')
        return format_html('<span style="color: orange;">✗ Unread</span>')

    read_status.short_description = "Read"

    def sent_status(self, obj):
        if obj.is_sent:
            return format_html('<span style="color: green;">✓ Sent</span>')
        return format_html('<span style="color: red;">✗ Not Sent</span>')

    sent_status.short_description = "Sent"

    actions = ["mark_as_read", "mark_as_sent"]

    def mark_as_read(self, request, queryset):
        updated = queryset.update(is_read=True)
        self.message_user(request, f"{updated} alert(s) marked as read.")

    mark_as_read.short_description = "Mark as read"

    def mark_as_sent(self, request, queryset):
        from django.utils import timezone

        updated = queryset.update(is_sent=True, sent_at=timezone.now())
        self.message_user(request, f"{updated} alert(s) marked as sent.")

    mark_as_sent.short_description = "Mark as sent"


@admin.register(PredictionLog)
class PredictionLogAdmin(admin.ModelAdmin):
    list_display = [
        "city_name",
        "prediction_date",
        "rain_prediction_display",
        "get_user_display",
        "models_used",
    ]
    list_filter = ["prediction_date", "rain_prediction"]
    search_fields = ["city_name", "user__username"]
    readonly_fields = ["prediction_date", "get_input_features_display"]
    date_hierarchy = "prediction_date"

    fieldsets = (
        ("Location & Time", {"fields": ("city_name", "prediction_date", "user")}),
        (
            "Models Used",
            {
                "fields": (
                    "rain_model_version",
                    "temp_model_version",
                    "humidity_model_version",
                )
            },
        ),
        (
            "Predictions",
            {
                "fields": (
                    "rain_prediction",
                    "rain_probability",
                    "temp_predictions",
                    "humidity_predictions",
                )
            },
        ),
        (
            "Confidence Intervals",
            {
                "fields": (
                    "temp_lower_bounds",
                    "temp_upper_bounds",
                    "humidity_lower_bounds",
                    "humidity_upper_bounds",
                ),
                "classes": ("collapse",),
            },
        ),
        (
            "Input Features",
            {"fields": ("get_input_features_display",), "classes": ("collapse",)},
        ),
    )

    def get_user_display(self, obj):
        if obj.user:
            return obj.user.username
        return "Anonymous"

    get_user_display.short_description = "User"

    def rain_prediction_display(self, obj):
        if obj.rain_prediction == 1:
            return format_html('<span style="color: blue;">☔ Rain</span>')
        return format_html('<span style="color: orange;">☀ No Rain</span>')

    rain_prediction_display.short_description = "Rain"

    def models_used(self, obj):
        return format_html(
            "R: {} | T: {} | H: {}",
            obj.rain_model_version,
            obj.temp_model_version,
            obj.humidity_model_version,
        )

    models_used.short_description = "Model Versions"

    def get_input_features_display(self, obj):
        import json

        try:
            features = json.loads(obj.input_features)
            return format_html("<pre>{}</pre>", json.dumps(features, indent=2))
        except:
            return obj.input_features

    get_input_features_display.short_description = "Input Features"


@admin.register(ModelComparisonResult)
class ModelComparisonResultAdmin(admin.ModelAdmin):
    list_display = [
        "model_type",
        "comparison_date",
        "model_a_name",
        "model_b_name",
        "winner_badge",
    ]
    list_filter = ["model_type", "winner", "comparison_date"]
    search_fields = ["model_a_name", "model_b_name", "notes"]
    readonly_fields = ["comparison_date"]
    date_hierarchy = "comparison_date"

    fieldsets = (
        ("Comparison Info", {"fields": ("comparison_date", "model_type", "winner")}),
        (
            "Model A",
            {
                "fields": (
                    "model_a_name",
                    "model_a_version",
                    "model_a_mae",
                    "model_a_rmse",
                    "model_a_r2",
                    "model_a_accuracy",
                )
            },
        ),
        (
            "Model B",
            {
                "fields": (
                    "model_b_name",
                    "model_b_version",
                    "model_b_mae",
                    "model_b_rmse",
                    "model_b_r2",
                    "model_b_accuracy",
                )
            },
        ),
        ("Notes", {"fields": ("notes",), "classes": ("collapse",)}),
    )

    def winner_badge(self, obj):
        colors = {
            "A": "#28a745",
            "B": "#007bff",
            "TIE": "#ffc107",
        }
        color = colors.get(obj.winner, "#6c757d")
        winner_text = obj.get_winner_display()
        return format_html(
            '<span style="background-color: {}; color: white; padding: 3px 10px; border-radius: 3px; font-weight: bold;">{}</span>',
            color,
            winner_text,
        )

    winner_badge.short_description = "Winner"
