import logging
import re
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class NewsFetcher:
    """Fetch location-specific news from Google News RSS feed"""

    def __init__(self):
        self.base_url = "https://news.google.com/rss/search"
        self.user_agent = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"

    def get_location_news(self, location, max_results=10):
        """
        Fetch top news articles related to a specific location

        Args:
            location (str): City or location name
            max_results (int): Maximum number of news articles to return

        Returns:
            list: List of news articles with title, link, published date, source
        """
        try:
            # Create search query for location-specific news
            query = f"{location} weather OR {location} news OR {location} today"
            encoded_query = urllib.parse.quote(query)

            # Build RSS URL
            rss_url = f"{self.base_url}?q={encoded_query}&hl=en-US&gl=US&ceid=US:en"

            # Fetch RSS feed
            req = urllib.request.Request(
                rss_url, headers={"User-Agent": self.user_agent}
            )

            with urllib.request.urlopen(req, timeout=10) as response:
                xml_data = response.read()

            # Parse XML
            root = ET.fromstring(xml_data)

            # Extract news items
            news_items = []
            channel = root.find("channel")

            if channel is None:
                logger.warning("No channel found in RSS feed")
                return []

            items = channel.findall("item")

            for item in items[:max_results]:
                try:
                    title = item.find("title")
                    link = item.find("link")
                    pub_date = item.find("pubDate")
                    source = item.find("source")

                    news_item = {
                        "title": title.text if title is not None else "No title",
                        "link": link.text if link is not None else "#",
                        "published": self._parse_date(
                            pub_date.text if pub_date is not None else None
                        ),
                        "source": source.text if source is not None else "Unknown",
                        "source_url": (
                            source.attrib.get("url", "#") if source is not None else "#"
                        ),
                    }

                    # Clean title (remove source attribution if present)
                    news_item["title"] = self._clean_title(news_item["title"])

                    news_items.append(news_item)

                except Exception as e:
                    logger.error(f"Error parsing news item: {str(e)}")
                    continue

            return news_items

        except urllib.error.URLError as e:
            logger.error(f"Network error fetching news: {str(e)}")
            return []
        except ET.ParseError as e:
            logger.error(f"XML parsing error: {str(e)}")
            return []
        except Exception as e:
            logger.error(f"Unexpected error fetching news: {str(e)}")
            return []

    def get_weather_news(self, location, max_results=10):
        """
        Fetch weather-specific news for a location (last 7 days only)

        Args:
            location (str): City or location name
            max_results (int): Maximum number of articles

        Returns:
            list: Weather-related news articles from the past week
        """
        try:
            # More specific query to get location-specific weather news only
            query = f'"{location}" weather'
            encoded_query = urllib.parse.quote(query)

            rss_url = f"{self.base_url}?q={encoded_query}&hl=en-US&gl=US&ceid=US:en"

            req = urllib.request.Request(
                rss_url, headers={"User-Agent": self.user_agent}
            )

            with urllib.request.urlopen(req, timeout=10) as response:
                xml_data = response.read()

            root = ET.fromstring(xml_data)
            news_items = []
            channel = root.find("channel")

            if channel is None:
                return []

            items = channel.findall("item")

            # Calculate date threshold (7 days ago)
            week_ago = datetime.now() - timedelta(days=7)

            for item in items:
                try:
                    title = item.find("title")
                    link = item.find("link")
                    pub_date = item.find("pubDate")
                    source = item.find("source")
                    description = item.find("description")

                    # Check if the news is actually about the searched location
                    title_text = title.text if title is not None else ""
                    description_text = (
                        description.text if description is not None else ""
                    )

                    # Filter: news must mention the location in title or description
                    if (
                        location.lower() not in title_text.lower()
                        and location.lower() not in description_text.lower()
                    ):
                        continue

                    # Parse the publication date
                    pub_date_str = pub_date.text if pub_date is not None else None
                    if pub_date_str:
                        try:
                            # Try to parse the date
                            pub_datetime = datetime.strptime(
                                pub_date_str, "%a, %d %b %Y %H:%M:%S %Z"
                            )
                        except ValueError:
                            try:
                                # Try alternative format with timezone
                                pub_datetime = datetime.strptime(
                                    pub_date_str, "%a, %d %b %Y %H:%M:%S %z"
                                )
                                # Remove timezone info for comparison
                                pub_datetime = pub_datetime.replace(tzinfo=None)
                            except ValueError:
                                # If parsing fails, skip date filtering for this item
                                pub_datetime = datetime.now()

                        # Skip news older than 7 days
                        if pub_datetime < week_ago:
                            continue

                    news_item = {
                        "title": title.text if title is not None else "No title",
                        "link": link.text if link is not None else "#",
                        "published": self._parse_date(pub_date_str),
                        "source": source.text if source is not None else "Unknown",
                        "description": self._clean_html(
                            description.text if description is not None else ""
                        ),
                    }

                    news_item["title"] = self._clean_title(news_item["title"])
                    news_items.append(news_item)

                    # Stop if we have enough results
                    if len(news_items) >= max_results:
                        break

                except Exception as e:
                    logger.error(f"Error parsing weather news item: {str(e)}")
                    continue

            return news_items

        except Exception as e:
            logger.error(f"Error fetching weather news: {str(e)}")
            return []

    def _parse_date(self, date_string):
        """
        Parse RSS date string to readable format

        Args:
            date_string (str): Date string from RSS

        Returns:
            str: Formatted date string
        """
        if not date_string:
            return "Unknown"

        try:
            # RSS date format: Wed, 13 Nov 2024 10:30:00 GMT
            dt = datetime.strptime(date_string, "%a, %d %b %Y %H:%M:%S %Z")
            return dt.strftime("%B %d, %Y at %I:%M %p")
        except ValueError:
            try:
                # Alternative format
                dt = datetime.strptime(date_string, "%a, %d %b %Y %H:%M:%S %z")
                return dt.strftime("%B %d, %Y at %I:%M %p")
            except ValueError:
                return date_string

    def _clean_title(self, title):
        """
        Clean news title by removing source attribution

        Args:
            title (str): Original title

        Returns:
            str: Cleaned title
        """
        if not title:
            return "No title"

        # Remove " - Source Name" pattern at the end
        title = re.sub(r"\s*-\s*[^-]+$", "", title)

        # Remove duplicate spaces
        title = re.sub(r"\s+", " ", title)

        return title.strip()

    def _clean_html(self, html_text):
        """
        Remove HTML tags from text

        Args:
            html_text (str): Text with HTML tags

        Returns:
            str: Plain text
        """
        if not html_text:
            return ""

        # Remove HTML tags
        clean = re.sub(r"<[^>]+>", "", html_text)

        # Decode HTML entities
        clean = clean.replace("&amp;", "&")
        clean = clean.replace("&lt;", "<")
        clean = clean.replace("&gt;", ">")
        clean = clean.replace("&quot;", '"')
        clean = clean.replace("&#39;", "'")

        # Remove extra whitespace
        clean = re.sub(r"\s+", " ", clean)

        return clean.strip()

    def get_trending_news(self, category="weather", max_results=10):
        """
        Get trending news in a specific category (last 7 days only)

        Args:
            category (str): News category
            max_results (int): Maximum articles

        Returns:
            list: Trending news articles from the past week
        """
        try:
            # Different RSS endpoints for trending topics
            if category == "weather":
                base = "https://news.google.com/rss/search?q=weather+forecast&hl=en-US&gl=US&ceid=US:en"
            else:
                base = f"https://news.google.com/rss/headlines/section/topic/{category.upper()}?hl=en-US&gl=US&ceid=US:en"

            req = urllib.request.Request(base, headers={"User-Agent": self.user_agent})

            with urllib.request.urlopen(req, timeout=10) as response:
                xml_data = response.read()

            root = ET.fromstring(xml_data)
            news_items = []
            channel = root.find("channel")

            if channel is None:
                return []

            items = channel.findall("item")

            # Calculate date threshold (7 days ago)
            week_ago = datetime.now() - timedelta(days=7)

            for item in items:
                try:
                    title = item.find("title")
                    link = item.find("link")
                    pub_date = item.find("pubDate")
                    source = item.find("source")

                    # Parse the publication date
                    pub_date_str = pub_date.text if pub_date is not None else None
                    if pub_date_str:
                        try:
                            # Try to parse the date
                            pub_datetime = datetime.strptime(
                                pub_date_str, "%a, %d %b %Y %H:%M:%S %Z"
                            )
                        except ValueError:
                            try:
                                # Try alternative format with timezone
                                pub_datetime = datetime.strptime(
                                    pub_date_str, "%a, %d %b %Y %H:%M:%S %z"
                                )
                                # Remove timezone info for comparison
                                pub_datetime = pub_datetime.replace(tzinfo=None)
                            except ValueError:
                                # If parsing fails, skip date filtering for this item
                                pub_datetime = datetime.now()

                        # Skip news older than 7 days
                        if pub_datetime < week_ago:
                            continue

                    news_item = {
                        "title": self._clean_title(
                            title.text if title is not None else "No title"
                        ),
                        "link": link.text if link is not None else "#",
                        "published": self._parse_date(pub_date_str),
                        "source": source.text if source is not None else "Unknown",
                    }

                    news_items.append(news_item)

                    # Stop if we have enough results
                    if len(news_items) >= max_results:
                        break

                except Exception as e:
                    logger.error(f"Error parsing trending news: {str(e)}")
                    continue

            return news_items

        except Exception as e:
            logger.error(f"Error fetching trending news: {str(e)}")
            return []


# Convenience function for easy import
def get_news_for_location(location, max_results=10):
    """
    Quick function to fetch news for a location

    Args:
        location (str): City name
        max_results (int): Number of articles

    Returns:
        list: News articles
    """
    fetcher = NewsFetcher()
    return fetcher.get_location_news(location, max_results)


def get_weather_news_for_location(location, max_results=10):
    """
    Quick function to fetch weather news for a location

    Args:
        location (str): City name
        max_results (int): Number of articles

    Returns:
        list: Weather news articles
    """
    fetcher = NewsFetcher()
    return fetcher.get_weather_news(location, max_results)
