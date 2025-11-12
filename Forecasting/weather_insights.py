"""
Intelligent Weather Insights & Actionable Suggestions
=====================================================
This module provides meaningful, actionable insights based on weather patterns:
- Smart clothing suggestions based on multiple factors
- Health risk assessments
- Optimal time recommendations for activities
- Travel advisories
- Agricultural insights
- Energy consumption predictions
- Sports & outdoor activity recommendations
- Photo opportunity alerts
- Severe weather preparedness tips
- Historical pattern matching
"""

from datetime import datetime, timedelta

import numpy as np
from django.db.models import Avg, Max, Min

from .models import PredictionLog, SearchHistory


class WeatherInsights:
    """Generate intelligent, actionable weather insights"""

    def get_comprehensive_insights(self, weather_data, forecast_data, city):
        """
        Generate comprehensive insights package
        Returns actionable suggestions across multiple categories
        """
        insights = {
            "clothing": self.get_clothing_recommendations(weather_data, forecast_data),
            "health": self.get_health_insights(weather_data),
            "activities": self.get_activity_recommendations(
                weather_data, forecast_data
            ),
            "travel": self.get_travel_insights(weather_data),
            "energy": self.get_energy_insights(weather_data),
            "agriculture": self.get_agriculture_insights(weather_data),
            "photography": self.get_photography_opportunities(weather_data),
            "best_times": self.get_optimal_times(forecast_data),
            "alerts": self.get_smart_alerts(weather_data, city),
            "comparison": self.get_historical_comparison(city, weather_data),
        }
        return insights

    def get_clothing_recommendations(self, weather_data, forecast_data):
        """
        Detailed clothing recommendations based on temperature, weather conditions,
        and time of day changes
        """
        temp = weather_data.get("current_temp", 20)
        description = weather_data.get("description", "").lower()
        humidity = weather_data.get("humidity", 50)
        wind_speed_kmh = weather_data.get("wind_speed", 0) * 3.6

        recommendations = {
            "morning": [],
            "afternoon": [],
            "evening": [],
            "accessories": [],
            "footwear": [],
            "special_notes": [],
        }

        # Temperature-based base layer
        if temp < 0:
            recommendations["morning"] = [
                "Heavy winter coat or parka",
                "Thermal underwear (base layer)",
                "Thick wool sweater",
                "Insulated pants",
            ]
            recommendations["accessories"] = [
                "Warm winter hat covering ears",
                "Insulated gloves or mittens",
                "Thick scarf or neck warmer",
                "Warm socks (wool or thermal)",
            ]
            recommendations["footwear"] = ["Insulated winter boots with good traction"]
        elif temp < 10:
            recommendations["morning"] = [
                "Winter jacket or heavy coat",
                "Long-sleeve shirt + sweater",
                "Long pants or jeans",
            ]
            recommendations["accessories"] = [
                "Light gloves",
                "Beanie or warm hat",
                "Scarf",
            ]
            recommendations["footwear"] = ["Closed shoes or boots"]
        elif temp < 15:
            recommendations["morning"] = [
                "Light jacket or hoodie",
                "Long-sleeve shirt",
                "Jeans or long pants",
            ]
            recommendations["afternoon"] = [
                "Can remove jacket if sunny",
                "Light sweater sufficient",
            ]
            recommendations["accessories"] = ["Light scarf (optional)"]
            recommendations["footwear"] = ["Sneakers or casual shoes"]
        elif temp < 20:
            recommendations["morning"] = ["Light sweater or cardigan", "Long pants"]
            recommendations["afternoon"] = [
                "T-shirt or short sleeves may be comfortable",
                "Light jacket for evening",
            ]
            recommendations["accessories"] = ["Sunglasses if sunny"]
            recommendations["footwear"] = ["Any comfortable shoes"]
        elif temp < 25:
            recommendations["morning"] = [
                "Light clothing",
                "T-shirt or blouse",
                "Jeans or light pants",
            ]
            recommendations["afternoon"] = ["Stay in light, breathable fabrics"]
            recommendations["accessories"] = ["Sunglasses", "Light cap or hat"]
            recommendations["footwear"] = ["Breathable shoes", "Sandals acceptable"]
        elif temp < 30:
            recommendations["morning"] = [
                "Light, breathable clothing",
                "T-shirt or tank top",
                "Shorts or light skirt",
            ]
            recommendations["accessories"] = [
                "Sunglasses (UV protection)",
                "Wide-brimmed hat",
                "Sunscreen SPF 30+",
            ]
            recommendations["footwear"] = ["Sandals or breathable shoes"]
            recommendations["special_notes"] = [
                "Stay hydrated",
                "Seek shade during peak hours",
            ]
        else:  # > 30°C
            recommendations["morning"] = [
                "Minimal, loose-fitting clothing",
                "Light colors (reflect heat)",
                "Moisture-wicking fabrics",
            ]
            recommendations["accessories"] = [
                "Essential: Sunglasses + Hat",
                "Sunscreen SPF 50+",
                "Carry water bottle",
            ]
            recommendations["footwear"] = ["Open sandals or breathable shoes"]
            recommendations["special_notes"] = [
                "HEAT WARNING: Limit outdoor exposure",
                "Stay in air-conditioned areas",
                "Drink water frequently",
            ]

        # Weather condition modifiers
        if "rain" in description or "drizzle" in description:
            recommendations["accessories"].insert(0, "🌂 Umbrella (essential)")
            recommendations["accessories"].append("Waterproof jacket or raincoat")
            recommendations["footwear"] = ["Waterproof shoes or rain boots"]
            recommendations["special_notes"].append(
                "Keep electronics protected from rain"
            )

        if wind_speed_kmh > 30:
            recommendations["special_notes"].append(
                f"Strong winds ({wind_speed_kmh:.0f} km/h): Secure loose clothing"
            )
            recommendations["accessories"].append("Windbreaker recommended")

        if humidity > 80:
            recommendations["special_notes"].append(
                "High humidity: Choose moisture-wicking fabrics"
            )

        return recommendations

    def get_health_insights(self, weather_data):
        """
        Health risk assessment and recommendations
        """
        temp = weather_data.get("current_temp", 20)
        humidity = weather_data.get("humidity", 50)
        description = weather_data.get("description", "").lower()
        wind_speed_kmh = weather_data.get("wind_speed", 0) * 3.6

        health_insights = {
            "risk_level": "low",
            "risks": [],
            "recommendations": [],
            "vulnerable_groups": [],
        }

        # Heat-related health risks
        if temp > 35:
            health_insights["risk_level"] = "high"
            health_insights["risks"].append(
                {
                    "type": "Heat Exhaustion Risk",
                    "severity": "high",
                    "symptoms": "Dizziness, nausea, headache, rapid heartbeat",
                    "action": "Stay indoors, drink water every 15 minutes, seek medical help if symptoms worsen",
                }
            )
            health_insights["vulnerable_groups"] = [
                "Elderly",
                "Young children",
                "Pregnant women",
                "People with heart conditions",
            ]

        elif temp > 30:
            health_insights["risk_level"] = "moderate"
            health_insights["risks"].append(
                {
                    "type": "Heat Stress",
                    "severity": "moderate",
                    "symptoms": "Excessive sweating, fatigue, thirst",
                    "action": "Drink 2-4 glasses of water per hour, take breaks in shade",
                }
            )

        # Cold-related health risks
        if temp < 0:
            health_insights["risk_level"] = "high"
            health_insights["risks"].append(
                {
                    "type": "Hypothermia & Frostbite Risk",
                    "severity": "high",
                    "symptoms": "Shivering, confusion, numbness in extremities",
                    "action": "Limit outdoor exposure, cover all exposed skin, seek warmth immediately if symptoms appear",
                }
            )
            health_insights["vulnerable_groups"] = [
                "Elderly",
                "Young children",
                "People with circulatory problems",
            ]

        elif temp < 5:
            health_insights["risk_level"] = "moderate"
            health_insights["risks"].append(
                {
                    "type": "Cold Stress",
                    "severity": "moderate",
                    "symptoms": "Shivering, cold hands and feet",
                    "action": "Dress in layers, keep moving, limit time outdoors",
                }
            )

        # Air quality and respiratory concerns
        if humidity > 85:
            health_insights["risks"].append(
                {
                    "type": "Respiratory Discomfort",
                    "severity": "low",
                    "symptoms": "Difficulty breathing, feeling stuffy",
                    "action": "People with asthma should carry inhalers, avoid strenuous outdoor activities",
                }
            )

        # General health recommendations
        if temp > 25 and humidity > 70:
            health_insights["recommendations"].append(
                "High heat + humidity = increased dehydration risk. Drink 3-4 liters of water today."
            )

        if "rain" in description:
            health_insights["recommendations"].append(
                "Rainy weather: Increased risk of slips and falls. Walk carefully on wet surfaces."
            )

        if wind_speed_kmh > 40:
            health_insights["recommendations"].append(
                "Strong winds: Risk of flying debris. Protect eyes, stay away from trees."
            )

        # Seasonal considerations
        month = datetime.now().month
        if month in [12, 1, 2] and temp < 10:
            health_insights["recommendations"].append(
                "Winter season: Boost immunity with vitamin C, get adequate sleep."
            )
        elif month in [6, 7, 8] and temp > 30:
            health_insights["recommendations"].append(
                "Summer heat: Eat light meals, avoid heavy exercise during midday."
            )

        return health_insights

    def get_activity_recommendations(self, weather_data, forecast_data):
        """
        Optimal activities based on weather conditions
        """
        temp = weather_data.get("current_temp", 20)
        description = weather_data.get("description", "").lower()
        wind_speed_kmh = weather_data.get("wind_speed", 0) * 3.6
        clouds = weather_data.get("clouds", 50)

        activities = {
            "highly_recommended": [],
            "suitable": [],
            "not_recommended": [],
            "indoor_alternatives": [],
        }

        # Perfect weather activities (15-25°C, low wind, partly cloudy)
        if 15 <= temp <= 25 and wind_speed_kmh < 20 and "rain" not in description:
            activities["highly_recommended"] = [
                {
                    "activity": "🚴 Cycling",
                    "reason": "Perfect temperature and conditions",
                    "best_time": "Morning or late afternoon",
                    "duration": "1-3 hours",
                },
                {
                    "activity": "🏃 Running/Jogging",
                    "reason": "Ideal temperature for cardiovascular exercise",
                    "best_time": "Early morning (6-9 AM)",
                    "duration": "30-60 minutes",
                },
                {
                    "activity": "🧺 Picnic",
                    "reason": "Comfortable outdoor conditions",
                    "best_time": "11 AM - 4 PM",
                    "duration": "2-4 hours",
                },
            ]

        # Hot weather activities
        if temp > 28 and "rain" not in description:
            activities["highly_recommended"].append(
                {
                    "activity": "🏊 Swimming",
                    "reason": "Perfect weather to cool off",
                    "best_time": "Afternoon (2-6 PM)",
                    "duration": "1-2 hours",
                }
            )
            activities["suitable"].append(
                {
                    "activity": "🏖️ Beach/Water Sports",
                    "reason": "Hot weather ideal for water activities",
                    "best_time": "Morning or late afternoon",
                }
            )

        # Cool weather activities
        if 5 <= temp <= 15 and "rain" not in description:
            activities["suitable"].extend(
                [
                    {
                        "activity": "🥾 Hiking",
                        "reason": "Cool weather reduces overheating risk",
                        "best_time": "10 AM - 3 PM (warmest part)",
                        "duration": "2-5 hours",
                    },
                    {
                        "activity": "📸 Photography Walk",
                        "reason": "Great lighting and comfortable temperature",
                        "best_time": "Early morning or late afternoon",
                    },
                ]
            )

        # Rainy day alternatives
        if "rain" in description or "drizzle" in description:
            activities["not_recommended"] = [
                "Outdoor sports",
                "Beach activities",
                "Hiking",
                "Outdoor events",
            ]
            activities["indoor_alternatives"] = [
                {
                    "activity": "🏛️ Museum Visit",
                    "reason": "Perfect rainy day activity",
                },
                {
                    "activity": "🎬 Movie Theater",
                    "reason": "Comfortable indoor entertainment",
                },
                {
                    "activity": "🏋️ Indoor Gym/Workout",
                    "reason": "Stay active without getting wet",
                },
                {
                    "activity": "☕ Café/Reading",
                    "reason": "Cozy indoor activity",
                },
                {
                    "activity": "🎳 Bowling/Indoor Games",
                    "reason": "Fun indoor social activity",
                },
            ]

        # Windy conditions
        if wind_speed_kmh > 30:
            activities["not_recommended"].extend(
                [
                    "Cycling (difficult to control)",
                    "Outdoor ball sports",
                    "Picnics",
                ]
            )

        # Extreme temperatures
        if temp > 35 or temp < 0:
            activities["not_recommended"].extend(
                [
                    "Prolonged outdoor exercise",
                    "Long-distance running",
                    "Outdoor sports",
                ]
            )
            activities["indoor_alternatives"].insert(
                0,
                {
                    "activity": "🏠 Indoor Activities Recommended",
                    "reason": "Extreme temperature conditions",
                },
            )

        # Clear sky activities
        if clouds < 30 and "rain" not in description:
            activities["suitable"].append(
                {
                    "activity": "⭐ Stargazing",
                    "reason": "Clear skies, perfect visibility",
                    "best_time": "After 9 PM",
                }
            )

        return activities

    def get_travel_insights(self, weather_data):
        """
        Travel advisories and recommendations
        """
        temp = weather_data.get("current_temp", 20)
        description = weather_data.get("description", "").lower()
        wind_speed_kmh = weather_data.get("wind_speed", 0) * 3.6
        visibility_km = weather_data.get("visibility", 10000) / 1000

        travel_insights = {
            "overall_conditions": "good",
            "driving_conditions": [],
            "flight_considerations": [],
            "public_transport": [],
            "recommendations": [],
        }

        # Visibility-based advisories
        if visibility_km < 1:
            travel_insights["overall_conditions"] = "poor"
            travel_insights["driving_conditions"].append(
                {
                    "severity": "high",
                    "issue": "Very low visibility (<1 km)",
                    "advice": "Avoid driving if possible. Use fog lights, reduce speed significantly.",
                }
            )
        elif visibility_km < 5:
            travel_insights["overall_conditions"] = "fair"
            travel_insights["driving_conditions"].append(
                {
                    "severity": "moderate",
                    "issue": "Reduced visibility",
                    "advice": "Drive carefully, use low beams, increase following distance.",
                }
            )

        # Weather condition advisories
        if "rain" in description or "drizzle" in description:
            travel_insights["driving_conditions"].append(
                {
                    "severity": "moderate",
                    "issue": "Wet roads",
                    "advice": "Reduce speed by 25%, increase braking distance, avoid sudden maneuvers.",
                }
            )
            travel_insights["public_transport"].append(
                "Expect delays. Roads may be congested. Plan extra 15-30 minutes."
            )

        if "thunderstorm" in description or "thunder" in description:
            travel_insights["overall_conditions"] = "poor"
            travel_insights["flight_considerations"].append(
                "⚠️ Flights may be delayed or cancelled due to thunderstorms. Check with airline."
            )
            travel_insights["recommendations"].append(
                "Delay non-essential travel. If driving, pull over safely during heavy storms."
            )

        if "snow" in description:
            travel_insights["overall_conditions"] = "poor"
            travel_insights["driving_conditions"].append(
                {
                    "severity": "high",
                    "issue": "Snow on roads",
                    "advice": "Use winter tires. Drive at least 50% slower. Keep emergency kit in car.",
                }
            )

        if wind_speed_kmh > 50:
            travel_insights["driving_conditions"].append(
                {
                    "severity": "moderate",
                    "issue": "Strong crosswinds",
                    "advice": "Hold steering wheel firmly, reduce speed on bridges and open areas.",
                }
            )
            travel_insights["flight_considerations"].append(
                "Expect turbulence. Flights may be delayed."
            )

        # Temperature-based advisories
        if temp < -5:
            travel_insights["driving_conditions"].append(
                {
                    "severity": "moderate",
                    "issue": "Risk of ice on roads",
                    "advice": "Watch for black ice especially on bridges. Start car earlier to warm up.",
                }
            )

        # Good conditions
        if (
            15 <= temp <= 28
            and visibility_km > 10
            and "rain" not in description
            and wind_speed_kmh < 20
        ):
            travel_insights["overall_conditions"] = "excellent"
            travel_insights["recommendations"].append(
                "✓ Excellent conditions for travel. Consider taking scenic routes!"
            )

        return travel_insights

    def get_energy_insights(self, weather_data):
        """
        Energy consumption predictions and cost-saving tips
        """
        temp = weather_data.get("current_temp", 20)
        humidity = weather_data.get("humidity", 50)
        description = weather_data.get("description", "").lower()

        energy_insights = {
            "expected_consumption": "normal",
            "cost_estimate": "moderate",
            "hvac_recommendations": [],
            "cost_saving_tips": [],
            "renewable_potential": [],
        }

        # Temperature-based energy consumption
        if temp > 30:
            energy_insights["expected_consumption"] = "high"
            energy_insights["cost_estimate"] = "20-40% above average"
            energy_insights["hvac_recommendations"] = [
                "Set AC to 24-25°C (optimal balance of comfort and efficiency)",
                "Use fans in conjunction with AC to circulate air",
                "Close curtains/blinds during peak sun hours (11 AM - 4 PM)",
                "Avoid using heat-generating appliances during day",
            ]
            energy_insights["cost_saving_tips"] = [
                "💡 Pre-cool home before peak electricity hours (usually 2-8 PM)",
                "💰 Run dishwasher and washing machine at night for lower rates",
                "🪟 Keep doors and windows closed when AC is running",
            ]
        elif temp < 5:
            energy_insights["expected_consumption"] = "high"
            energy_insights["cost_estimate"] = "30-50% above average"
            energy_insights["hvac_recommendations"] = [
                "Set heating to 18-20°C during day, 15-17°C at night",
                "Use space heaters only in occupied rooms",
                "Keep doors closed to trap heat in specific areas",
                "Ensure windows and doors are properly sealed",
            ]
            energy_insights["cost_saving_tips"] = [
                "🧥 Wear layers instead of increasing heating",
                "☕ Use hot beverages to feel warmer",
                "🛏️ Use electric blankets (more efficient than heating entire room)",
            ]
        else:
            energy_insights["expected_consumption"] = "normal"
            energy_insights["cost_estimate"] = "average"
            energy_insights["cost_saving_tips"] = [
                "🌡️ Open windows for natural ventilation",
                "💡 Take advantage of natural daylight",
                "🔌 Unplug devices not in use (phantom power drain)",
            ]

        # Solar potential
        if "clear" in description or "sun" in description:
            energy_insights["renewable_potential"].append(
                {
                    "type": "Solar",
                    "potential": "high",
                    "note": "Excellent day for solar energy generation. Solar panels at peak efficiency.",
                }
            )

        # Wind potential
        wind_speed_kmh = weather_data.get("wind_speed", 0) * 3.6
        if wind_speed_kmh > 15:
            energy_insights["renewable_potential"].append(
                {
                    "type": "Wind",
                    "potential": "good",
                    "note": f"Wind speed {wind_speed_kmh:.0f} km/h suitable for wind energy generation.",
                }
            )

        return energy_insights

    def get_agriculture_insights(self, weather_data):
        """
        Agricultural insights and recommendations
        """
        temp = weather_data.get("current_temp", 20)
        humidity = weather_data.get("humidity", 50)
        description = weather_data.get("description", "").lower()
        rainfall = weather_data.get("rainfall", 0)

        ag_insights = {
            "irrigation_needed": False,
            "planting_conditions": [],
            "crop_care": [],
            "pest_warnings": [],
            "harvest_recommendations": [],
        }

        # Irrigation recommendations
        if rainfall > 5:
            ag_insights["irrigation_needed"] = False
            ag_insights["crop_care"].append(
                "✓ Sufficient rainfall. Skip irrigation today."
            )
        elif temp > 30 and humidity < 40:
            ag_insights["irrigation_needed"] = True
            ag_insights["crop_care"].append(
                "🚰 HIGH PRIORITY: Irrigate crops early morning or evening. Hot and dry conditions increase water stress."
            )
        elif humidity < 50 and rainfall == 0:
            ag_insights["irrigation_needed"] = True
            ag_insights["crop_care"].append(
                "🚰 Moderate irrigation needed. Water in early morning."
            )

        # Planting conditions
        if 15 <= temp <= 30 and humidity > 40 and "rain" not in description:
            ag_insights["planting_conditions"].append(
                {
                    "suitability": "excellent",
                    "note": "Ideal conditions for planting. Soil is workable and temperature is optimal.",
                    "best_for": "Vegetables, herbs, flowering plants",
                }
            )
        elif temp < 10:
            ag_insights["planting_conditions"].append(
                {
                    "suitability": "poor",
                    "note": "Too cold for most plants. Risk of frost damage.",
                    "action": "Delay planting or use greenhouse",
                }
            )

        # Frost warning
        if temp < 4 and temp > 0:
            ag_insights["crop_care"].append(
                "⚠️ FROST WARNING: Cover sensitive plants. Bring potted plants indoors."
            )
        elif temp <= 0:
            ag_insights["crop_care"].append(
                "❄️ FREEZE WARNING: Serious risk to crops. Implement frost protection immediately."
            )

        # Pest and disease warnings
        if humidity > 80 and 20 <= temp <= 30:
            ag_insights["pest_warnings"].append(
                {
                    "risk": "Fungal diseases",
                    "severity": "high",
                    "action": "Avoid overhead watering. Ensure good air circulation. Apply fungicide if needed.",
                }
            )

        if temp > 25 and humidity < 50:
            ag_insights["pest_warnings"].append(
                {
                    "risk": "Spider mites, aphids",
                    "severity": "moderate",
                    "action": "Monitor plants closely. Hot, dry conditions favor these pests.",
                }
            )

        # Harvest timing
        if "rain" in description:
            ag_insights["harvest_recommendations"].append(
                "⚠️ Delay harvest if possible. Wet conditions can damage crops and reduce storage life."
            )
        elif 15 <= temp <= 28 and humidity < 70 and "clear" in description:
            ag_insights["harvest_recommendations"].append(
                "✓ Excellent harvest conditions. Crops will be dry and easy to handle."
            )

        return ag_insights

    def get_photography_opportunities(self, weather_data):
        """
        Photography opportunities and best timing
        """
        description = weather_data.get("description", "").lower()
        clouds = weather_data.get("clouds", 50)
        hour = datetime.now().hour

        photo_ops = {
            "golden_hour": False,
            "blue_hour": False,
            "opportunities": [],
            "best_subjects": [],
            "settings_recommendations": [],
        }

        # Golden hour (warm, soft light)
        if 5 <= hour <= 7 or 17 <= hour <= 19:
            photo_ops["golden_hour"] = True
            photo_ops["opportunities"].append(
                {
                    "type": "Golden Hour",
                    "quality": "excellent",
                    "description": "Warm, directional light perfect for portraits and landscapes",
                    "timing": "Next 1-2 hours"
                    if hour in [5, 6, 17, 18]
                    else "Soon (check exact times)",
                }
            )

        # Blue hour (cool, soft light)
        if 4 <= hour <= 5 or 19 <= hour <= 20:
            photo_ops["blue_hour"] = True
            photo_ops["opportunities"].append(
                {
                    "type": "Blue Hour",
                    "quality": "excellent",
                    "description": "Beautiful blue tones, perfect for cityscapes and architecture",
                    "timing": "Current" if hour in [4, 19] else "Soon",
                }
            )

        # Cloud-based opportunities
        if 20 <= clouds <= 60:
            photo_ops["opportunities"].append(
                {
                    "type": "Dramatic Sky",
                    "quality": "very good",
                    "description": "Partial cloud cover creates interesting textures and patterns",
                    "best_subjects": ["Landscapes", "Seascapes", "Long exposures"],
                }
            )

        if clouds > 80 and "rain" not in description:
            photo_ops["opportunities"].append(
                {
                    "type": "Overcast Light",
                    "quality": "good",
                    "description": "Soft, diffused light perfect for portraits and macro",
                    "best_subjects": [
                        "Portraits",
                        "Flowers",
                        "Nature details",
                        "Street photography",
                    ],
                }
            )
            photo_ops["settings_recommendations"].append(
                "Use wider aperture (f/2.8-f/5.6) to compensate for lower light"
            )

        if "rain" in description or "drizzle" in description:
            photo_ops["opportunities"].append(
                {
                    "type": "Rain Photography",
                    "quality": "unique",
                    "description": "Reflections in puddles, rain drops, moody atmosphere",
                    "best_subjects": [
                        "Reflections",
                        "City lights",
                        "Rain drops on glass",
                    ],
                }
            )
            photo_ops["settings_recommendations"].extend(
                [
                    "Protect camera with weather-sealed bag or cover",
                    "Use faster shutter speeds (1/500s+) to freeze rain drops",
                    "Try long exposures (1-4s) for rain streaks effect",
                ]
            )

        if "clear" in description and clouds < 20:
            if 10 <= hour <= 14:
                photo_ops["opportunities"].append(
                    {
                        "type": "Harsh Midday Light",
                        "quality": "challenging",
                        "description": "Strong shadows and high contrast",
                        "best_subjects": [
                            "Architecture (shadows)",
                            "Abstract patterns",
                        ],
                        "note": "Not ideal for portraits. Better opportunities in 2-3 hours.",
                    }
                )
            else:
                photo_ops["opportunities"].append(
                    {
                        "type": "Clear Sky",
                        "quality": "good",
                        "description": "Clean, clear light with strong colors",
                        "best_subjects": ["Landscapes", "Architecture", "Travel"],
                    }
                )

        if "fog" in description or "mist" in description:
            photo_ops["opportunities"].append(
                {
                    "type": "Atmospheric Fog",
                    "quality": "excellent",
                    "description": "Mysterious, moody scenes with depth and layers",
                    "best_subjects": [
                        "Landscapes",
                        "Forests",
                        "Street scenes",
                        "Silhouettes",
                    ],
                }
            )

        return photo_ops

    def get_optimal_times(self, forecast_data):
        """
        Determine best times for various activities based on forecast
        """
        if not forecast_data:
            return None

        optimal_times = {
            "exercise": None,
            "outdoor_dining": None,
            "commute": None,
            "laundry": None,
            "car_wash": None,
        }

        # Analyze forecast (simplified - assumes forecast is list of hourly data)
        # Find coolest hour for exercise
        # Find clearest weather for outdoor activities
        # etc.

        optimal_times["exercise"] = {
            "recommended": "Early morning (6-8 AM) or evening (5-7 PM)",
            "reason": "Cooler temperatures, less UV exposure",
        }

        optimal_times["outdoor_dining"] = {
            "recommended": "Evening (6-8 PM)",
            "reason": "Pleasant temperature, less direct sunlight",
        }

        return optimal_times

    def get_smart_alerts(self, weather_data, city):
        """
        Intelligent, context-aware alerts
        """
        alerts = []

        temp = weather_data.get("current_temp", 20)
        description = weather_data.get("description", "").lower()
        pressure = weather_data.get("pressure", 1013)

        # Rapid pressure drop = storm coming
        if pressure < 995:
            alerts.append(
                {
                    "priority": "high",
                    "icon": "⛈️",
                    "title": "Storm Warning",
                    "message": "Very low pressure detected. Storm likely within 12-24 hours.",
                    "actions": [
                        "Secure outdoor items",
                        "Charge devices",
                        "Check emergency supplies",
                    ],
                }
            )

        # Temperature anomaly
        if temp > 35:
            alerts.append(
                {
                    "priority": "high",
                    "icon": "🌡️",
                    "title": "Extreme Heat",
                    "message": f"Temperature is {temp}°C. Heat exhaustion risk is high.",
                    "actions": [
                        "Stay hydrated",
                        "Avoid outdoor activities",
                        "Check on vulnerable people",
                    ],
                }
            )

        # Air quality (if implemented)
        # Pollen levels (if implemented)
        # UV index warnings

        return alerts

    def get_historical_comparison(self, city, current_weather):
        """
        Compare current weather to historical patterns
        """
        try:
            # Get historical predictions for this city
            month = datetime.now().month
            historical = PredictionLog.objects.filter(
                city_name__iexact=city, prediction_date__month=month
            ).aggregate(
                avg_temp=Avg("input_features__Temp"),
                max_temp=Max("input_features__Temp"),
                min_temp=Min("input_features__Temp"),
            )

            if historical["avg_temp"]:
                current_temp = current_weather.get("current_temp", 20)
                avg_temp = float(historical["avg_temp"])

                comparison = {
                    "current": current_temp,
                    "historical_average": round(avg_temp, 1),
                    "difference": round(current_temp - avg_temp, 1),
                    "status": "typical",
                }

                if abs(current_temp - avg_temp) > 5:
                    if current_temp > avg_temp:
                        comparison["status"] = "warmer than usual"
                        comparison["note"] = (
                            f"It's {abs(current_temp - avg_temp):.1f}°C warmer than average for this time."
                        )
                    else:
                        comparison["status"] = "cooler than usual"
                        comparison["note"] = (
                            f"It's {abs(current_temp - avg_temp):.1f}°C cooler than average for this time."
                        )
                else:
                    comparison["note"] = "Temperature is typical for this time of year."

                return comparison
        except Exception as e:
            print(f"Error in historical comparison: {e}")

        return None
