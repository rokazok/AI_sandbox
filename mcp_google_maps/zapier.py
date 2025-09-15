import os
import requests
import argparse
from dotenv import load_dotenv

load_dotenv()
GOOGLE_MAPS_API_KEY = os.getenv("GOOGLE_MAPS_API_KEY")

def geocode_address(address: str) -> tuple[float, float] | None:
    """
    Geocode an address or place name to latitude and longitude using Google Geocoding API.

    Args:
        address (str): The address, neighborhood, zip code, or landmark.

    Returns:
        tuple: (latitude, longitude) or None if not found.
    """
    url = "https://maps.googleapis.com/maps/api/geocode/json"
    params = {
        "address": address,
        "key": GOOGLE_MAPS_API_KEY
    }
    try:
        response = requests.get(url, params=params, timeout=15)
        response.raise_for_status()
        result = response.json()
        if result.get("results"):
            location = result["results"][0]["geometry"]["location"]
            return location["lat"], location["lng"]
    except Exception:
        pass
    return None

def search_restaurants(
    latitude: float,
    longitude: float,
    radius: int = 5000,
    cuisine: str = None,
    max_results: int = 5,
    min_rating: float = 4.5
) -> str:
    """
    Search for restaurants near a location using Google Maps Places API, sorted by weighted rating.

    Args:
        latitude (float): Latitude of the center point.
        longitude (float): Longitude of the center point.
        radius (int, optional): Search radius in meters. Default is 1000 meters.
        cuisine (str, optional): Filter by cuisine type (e.g., 'seafood', 'italian'). Default is None (no filter).
        max_results (int, optional): Maximum number of results to return. Default is 5.
        min_rating (float, optional): Minimum average rating to include. Default is 4.5.

    Returns:
        str: Formatted list of restaurants or error message.
    """
    url = "https://places.googleapis.com/v1/places:searchNearby"
    headers = {
        "Content-Type": "application/json",
        "X-Goog-Api-Key": GOOGLE_MAPS_API_KEY,
        "X-Goog-FieldMask": "places.displayName,places.rating,places.userRatingCount,places.priceLevel,places.editorialSummary,places.types"
    }
    data = {
        "includedTypes": ["restaurant"],
        "maxResultCount": 20,  # Get more to allow for sorting and filtering
        "locationRestriction": {
            "circle": {
                "center": {"latitude": latitude, "longitude": longitude},
                "radius": radius
            }
        }
    }
    try:
        response = requests.post(url, headers=headers, json=data, timeout=15)
        response.raise_for_status()
        result = response.json()
        if not result or "places" not in result:
            return "No restaurants found or API error."
        places = result["places"]
        filtered = []
        ratings = []
        rating_counts = []

        # Filter by cuisine, min_rating, and collect ratings/counts
        for place in places:
            name = place.get("displayName", {}).get("text", "N/A")
            rating = place.get("rating", 0)
            v = place.get("userRatingCount", 0)
            price_level = place.get("priceLevel", "N/A")
            summary = place.get("editorialSummary", {}).get("text", "N/A")
            types = place.get("types", [])
            if cuisine and cuisine.lower() not in summary.lower() and cuisine.lower() not in str(types).lower():
                continue
            if rating < min_rating:
                continue
            filtered.append({
                "name": name,
                "rating": rating,
                "v": v,
                "price_level": price_level,
                "summary": summary,
                "types": types
            })
            ratings.append(rating)
            rating_counts.append(v)

        if not filtered:
            return "No matching restaurants found."

        # Calculate C (mean rating) and m (median rating count)
        C = sum(ratings) / len(ratings) if ratings else 0
        m = int(sorted(rating_counts)[len(rating_counts)//2]) if rating_counts else 0

        # Compute weighted rating for each restaurant
        for place in filtered:
            R = place["rating"]
            v = place["v"]
            # Weighted rating formula
            weighted = (R * v + C * m) / (v + m) if (v + m) > 0 else 0
            place["weighted_rating"] = weighted

        # Sort by weighted rating, descending
        filtered.sort(key=lambda x: x["weighted_rating"], reverse=True)

        # Format output
        output = []
        for place in filtered[:max_results]:
            output.append(
                f"Name: {place['name']}\n"
                f"Weighted Rating: {place['weighted_rating']:.2f}\n"
                f"Rating: {place['rating']} (based on {place['v']} reviews)\n"
                f"Price Level: {place['price_level']}\n"
                f"Summary: {place['summary']}\n"
            )
        return "\n---\n".join(output)
    except Exception as e:
        return f"No restaurants found or API error: {e}"

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Find top restaurants near an address using Google Maps API.")
    parser.add_argument("address", type=str, help="The address, neighborhood, zip code, or landmark to search near.")
    parser.add_argument("--radius", type=int, default=5000, help="Search radius in meters (default: 5000).")
    parser.add_argument("--cuisine", type=str, default=None, help="Cuisine type to filter by (optional).")
    parser.add_argument("--max_results", type=int, default=6, help="Maximum number of results to return (default: 6).")
    parser.add_argument("--min_rating", type=float, default=4.5, help="Minimum average rating to include (default: 4.5).")
    args = parser.parse_args()

    coords = geocode_address(args.address)
    if coords:
        print(f"Coordinates for '{args.address}': {coords}")
        print(search_restaurants(
            coords[0],
            coords[1],
            radius=args.radius,
            cuisine=args.cuisine,
            max_results=args.max_results,
            min_rating=args.min_rating
        ))
    else:
        print("Could not geocode the address.")