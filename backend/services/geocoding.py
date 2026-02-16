from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut, GeocoderUnavailable
import time

geolocator = Nominatim(user_agent="smarttrip_app_v1", timeout=10)

def get_coordinates(city: str):

    city = city.strip()

    try:
        location = geolocator.geocode(city)

        if location:
            return (location.latitude, location.longitude)

        # fallback
        location = geolocator.geocode(f"{city}, Europe")
        if location:
            return (location.latitude, location.longitude)

        return None

    except (GeocoderTimedOut, GeocoderUnavailable):
        time.sleep(1)
        return None
