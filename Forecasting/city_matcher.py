"""
City Name Matcher
=================
Provides fuzzy matching for city names to handle typos and suggest corrections
"""

from difflib import SequenceMatcher, get_close_matches


class CityMatcher:
    """Handles city name matching and typo correction"""

    # Common cities database (expandable)
    COMMON_CITIES = [
        # Nepal
        "Kathmandu",
        "Pokhara",
        "Lalitpur",
        "Bharatpur",
        "Biratnagar",
        "Birgunj",
        "Dharan",
        "Hetauda",
        "Janakpur",
        "Butwal",
        "Dhangadhi",
        "Itahari",
        "Bhaktapur",
        "Nepalgunj",
        "Banepa",
        "Gorkha",
        "Baglung",
        "Banepā",
        "Dolakha",
        "Damak",
        # India
        "Delhi",
        "Mumbai",
        "Bangalore",
        "Kolkata",
        "Chennai",
        "Hyderabad",
        "Pune",
        "Ahmedabad",
        "Jaipur",
        "Lucknow",
        "Chandigarh",
        "Patna",
        "Agra",
        "Varanasi",
        # Other Asian Countries
        "Bangkok",
        "Singapore",
        "Tokyo",
        "Seoul",
        "Beijing",
        "Shanghai",
        "Hong Kong",
        "Kuala Lumpur",
        "Jakarta",
        "Manila",
        "Dhaka",
        "Colombo",
        "Thimphu",
        # Europe
        "London",
        "Paris",
        "Berlin",
        "Madrid",
        "Rome",
        "Amsterdam",
        "Vienna",
        "Prague",
        "Oslo",
        "Stockholm",
        "Copenhagen",
        "Brussels",
        "Zurich",
        "Athens",
        "Lisbon",
        "Dublin",
        "Warsaw",
        "Budapest",
        "Norway",
        # North America
        "New York",
        "Los Angeles",
        "Chicago",
        "Houston",
        "Toronto",
        "Vancouver",
        "Montreal",
        "Mexico City",
        "San Francisco",
        "Boston",
        "Seattle",
        "Miami",
        "Las Vegas",
        # Middle East
        "Dubai",
        "Abu Dhabi",
        "Doha",
        "Riyadh",
        "Kuwait City",
        "Muscat",
        "Beirut",
        "Jerusalem",
        "Amman",
        "Istanbul",
        # Australia & Oceania
        "Sydney",
        "Melbourne",
        "Brisbane",
        "Perth",
        "Auckland",
        "Wellington",
        # South America
        "São Paulo",
        "Rio de Janeiro",
        "Buenos Aires",
        "Lima",
        "Bogotá",
        "Santiago",
        "Brasília",
        # Africa
        "Cairo",
        "Lagos",
        "Nairobi",
        "Johannesburg",
        "Cape Town",
        "Casablanca",
        "Addis Ababa",
        # Countries (that users might search as cities)
        "Thailand",
        "Malaysia",
        "Indonesia",
        "Vietnam",
        "Myanmar",
        "Cambodia",
        "Laos",
    ]

    @classmethod
    def find_closest_match(cls, input_city: str, cutoff: float = 0.6) -> dict:
        """
        Find the closest matching city name

        Args:
            input_city: The city name entered by user (possibly with typos)
            cutoff: Similarity threshold (0.0 to 1.0)

        Returns:
            dict with 'matched', 'original', 'suggestion', and 'confidence'
        """
        if not input_city or len(input_city.strip()) < 2:
            return {
                "matched": False,
                "original": input_city,
                "suggestion": None,
                "confidence": 0.0,
            }

        input_city = input_city.strip()

        # Try exact match first (case-insensitive)
        for city in cls.COMMON_CITIES:
            if city.lower() == input_city.lower():
                return {
                    "matched": True,
                    "original": input_city,
                    "suggestion": city,
                    "confidence": 1.0,
                }

        # Try fuzzy matching
        matches = get_close_matches(input_city, cls.COMMON_CITIES, n=1, cutoff=cutoff)

        if matches:
            best_match = matches[0]
            confidence = cls._calculate_similarity(input_city, best_match)

            return {
                "matched": True,
                "original": input_city,
                "suggestion": best_match,
                "confidence": confidence,
            }

        # No close match found
        return {
            "matched": False,
            "original": input_city,
            "suggestion": None,
            "confidence": 0.0,
        }

    @classmethod
    def get_suggestions(cls, input_city: str, n: int = 3, cutoff: float = 0.5) -> list:
        """
        Get multiple city suggestions

        Args:
            input_city: The city name entered by user
            n: Number of suggestions to return
            cutoff: Similarity threshold

        Returns:
            List of suggested city names with confidence scores
        """
        if not input_city or len(input_city.strip()) < 2:
            return []

        input_city = input_city.strip()

        # Get multiple matches
        matches = get_close_matches(input_city, cls.COMMON_CITIES, n=n, cutoff=cutoff)

        suggestions = []
        for match in matches:
            confidence = cls._calculate_similarity(input_city, match)
            suggestions.append({"city": match, "confidence": confidence})

        return suggestions

    @staticmethod
    def _calculate_similarity(str1: str, str2: str) -> float:
        """
        Calculate similarity ratio between two strings

        Args:
            str1: First string
            str2: Second string

        Returns:
            Similarity ratio (0.0 to 1.0)
        """
        return SequenceMatcher(None, str1.lower(), str2.lower()).ratio()

    @classmethod
    def correct_city_name(cls, input_city: str, auto_correct_threshold: float = 0.85):
        """
        Automatically correct city name if confidence is high enough

        Args:
            input_city: The city name entered by user
            auto_correct_threshold: Minimum confidence to auto-correct

        Returns:
            tuple: (corrected_city, was_corrected, suggestions)
        """
        result = cls.find_closest_match(input_city)

        if not result["matched"]:
            # No match found, try with lower threshold for suggestions
            suggestions = cls.get_suggestions(input_city, n=3, cutoff=0.4)
            return input_city, False, suggestions

        # High confidence - auto-correct
        if result["confidence"] >= auto_correct_threshold:
            return result["suggestion"], True, []

        # Medium confidence - provide suggestions
        suggestions = cls.get_suggestions(input_city, n=3)
        return input_city, False, suggestions
