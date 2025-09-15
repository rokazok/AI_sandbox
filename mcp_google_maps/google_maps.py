from typing import Any
import os
import httpx
from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP
from datetime import datetime, timedelta

load_dotenv()
GOOGLE_MAPS_API_KEY = os.getenv("GOOGLE_MAPS_API_KEY")

mcp = FastMCP("google_maps")

# --- Helper functions ---
async def async_post(url: str, headers: dict, data: dict) -> dict[str, Any] | None:
    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(url, headers=headers, json=data, timeout=30.0)
            response.raise_for_status()
            return response.json()
        except Exception:
            return None

async def async_get(url: str, headers: dict = None) -> dict[str, Any] | None:
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(url, headers=headers, timeout=30.0)
            response.raise_for_status()
            return response.json()
        except Exception:
            return None

# --- MCP Tools ---
@mcp.tool()
async def search_restaurants(
    latitude: float,
    longitude: float,
    radius: int = 1000,
    cuisine: str = None,
    max_results: int = 5
) -> str:
    """
    Search for restaurants near a location using Google Maps Places API.

    Args:
        latitude (float): Latitude of the center point.
        longitude (float): Longitude of the center point.
        radius (int, optional): Search radius in meters. Default is 1000 meters.
        cuisine (str, optional): Filter by cuisine type (e.g., 'seafood', 'italian'). Default is None (no filter).
        max_results (int, optional): Maximum number of results to return. Default is 5.

    Returns:
        str: Formatted list of restaurants or error message.
    """
    url = "https://places.googleapis.com/v1/places:searchNearby"
    headers = {
        "Content-Type": "application/json",
        "X-Goog-Api-Key": GOOGLE_MAPS_API_KEY,
        "X-Goog-FieldMask": "places.displayName,places.rating,places.priceLevel,places.editorialSummary,places.types"
    }
    data = {
        "includedTypes": ["restaurant"],
        "maxResultCount": max_results,
        "locationRestriction": {
            "circle": {
                "center": {"latitude": latitude, "longitude": longitude},
                "radius": radius
            }
        }
    }
    result = await async_post(url, headers, data)
    if not result or "places" not in result:
        return "No restaurants found or API error."
    places = result["places"]
    filtered = []
    for place in places:
        name = place.get("displayName", {}).get("text", "N/A")
        rating = place.get("rating", "N/A")
        price_level = place.get("priceLevel", "N/A")
        summary = place.get("editorialSummary", {}).get("text", "N/A")
        types = place.get("types", [])
        if cuisine and cuisine.lower() not in summary.lower() and cuisine.lower() not in str(types).lower():
            continue
        filtered.append(f"Name: {name}\nRating: {rating}\nPrice Level: {price_level}\nSummary: {summary}\n")
    if not filtered:
        return "No matching restaurants found."
    return "\n---\n".join(filtered)

@mcp.tool()
async def get_directions(
    origin_lat: float,
    origin_lng: float,
    dest_lat: float,
    dest_lng: float
) -> str:
    """
    Get driving directions between two points using Google Maps Routes API.

    Args:
        origin_lat (float): Latitude of the origin point.
        origin_lng (float): Longitude of the origin point.
        dest_lat (float): Latitude of the destination point.
        dest_lng (float): Longitude of the destination point.

    Returns:
        str: Route duration and distance, or error message.
    """
    url = "https://routes.googleapis.com/directions/v2:computeRoutes"
    headers = {
        "Content-Type": "application/json",
        "X-Goog-Api-Key": GOOGLE_MAPS_API_KEY,
        "X-Goog-FieldMask": "routes.duration,routes.distanceMeters,routes.polyline.encodedPolyline"
    }
    data = {
        "origin": {"location": {"latLng": {"latitude": origin_lat, "longitude": origin_lng}}},
        "destination": {"location": {"latLng": {"latitude": dest_lat, "longitude": dest_lng}}},
        "travelMode": "DRIVE",
        "routingPreference": "TRAFFIC_AWARE"
    }
    result = await async_post(url, headers, data)
    if not result or "routes" not in result or not result["routes"]:
        return "No route found or API error."
    route = result["routes"][0]
    duration = route.get("duration", "N/A")
    distance = route.get("distanceMeters", "N/A")
    return f"Route Duration: {duration}\nDistance: {distance} meters"

@mcp.tool()
async def get_weather(
    latitude: float,
    longitude: float,
    start: str = None,
    end: str = None
) -> str:
    """
    Get a next 10 days weather forecast for a location using Google Weather API.

    Args:
        latitude (float): Latitude of the location.
        longitude (float): Longitude of the location.
        start (str, optional): Start datetime in ISO format (YYYY-MM-DD or YYYY-MM-DD HH:MM). If None, start is today at 00:00.
        end (str, optional): End datetime in ISO format (YYYY-MM-DD or YYYY-MM-DD HH:MM). If None, end is today + 10 days at 23:59.

    Returns:
        str: Formatted table of weather forecast for the specified timeframe.
    """
    def parse_datetime(dt_str: str, is_start: bool) -> datetime:
        if not dt_str:
            return None
        try:
            # Try full datetime first
            return datetime.strptime(dt_str, "%Y-%m-%d %H:%M")
        except ValueError:
            try:
                # Fallback to date only
                date = datetime.strptime(dt_str, "%Y-%m-%d")
                if is_start:
                    return date.replace(hour=0, minute=0)
                else:
                    return date.replace(hour=23, minute=59)
            except ValueError:
                return None

    today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    ten_days_later = today + timedelta(days=10)

    start_datetime = parse_datetime(start, is_start=True) if start else today
    end_datetime = parse_datetime(end, is_start=False) if end else ten_days_later.replace(hour=23, minute=59)

    if start_datetime > ten_days_later or end_datetime < today:
        return f"Requested interval {start_datetime} - {end_datetime} does not overlap with the next 10 days from {today} - {ten_days_later}."

    url = (
        f"https://weather.googleapis.com/v1/forecast/hours:lookup"
        f"?key={GOOGLE_MAPS_API_KEY}"
        f"&location.latitude={latitude}&location.longitude={longitude}"
        f"&hours=240&pageSize=240"
    )
    result = await async_get(url)
    if not result or "forecastHours" not in result:
        return "Weather API error or no forecast data available."

    # Convert temperature to Fahrenheit
    def c_to_f(celsius):
        if celsius == "N/A":
            return "N/A"
        return round((float(celsius) * 9/5) + 32, 1)

    # Table header
    output = ["DateTime | Weather | Temp°F | Feels Like | Rain% | Wind | Humidity"]
    output.append("-" * 75)  # Separator line

    forecast_hours = result["forecastHours"]
    for hour in forecast_hours:
        dt = hour.get("displayDateTime", {})
        date_str = f"{dt.get('year', '')}-{dt.get('month', ''):02d}-{dt.get('day', ''):02d} {dt.get('hours', ''):02d}:00"
        
        # Skip if outside requested timeframe
        if start_datetime and date_str < start_datetime:
            continue
        if end_datetime and date_str > end_datetime:
            continue

        weather = hour.get("weatherCondition", {}).get("type", "N/A")
        temp = c_to_f(hour.get("temperature", {}).get("degrees", "N/A"))
        feels_like = c_to_f(hour.get("feelsLikeTemperature", {}).get("degrees", "N/A"))
        precip = hour.get("precipitation", {}).get("probability", {}).get("percent", 0)
        
        wind = hour.get("wind", {})
        wind_speed = wind.get("speed", {}).get("value", "N/A")
        wind_dir = wind.get("direction", {}).get("cardinal", "")
        wind_str = f"{wind_speed}km/h {wind_dir}"
        
        humidity = hour.get("relativeHumidity", "N/A")

        # Format each row with consistent spacing
        row = (
            f"{date_str} | "
            f"{weather:<12} | "
            f"{temp:>5} | "
            f"{feels_like:>9} | "
            f"{precip:>4}% | "
            f"{wind_str:<12} | "
            f"{humidity}%"
        )
        output.append(row)

    if len(output) <= 2:  # Only header and separator
        return "No weather data available for the specified timeframe."

    return "\n".join(output)

# @mcp.tool()
# async def render_static_route_map(
#     origin_lat: float,
#     origin_lng: float,
#     dest_lat: float,
#     dest_lng: float,
#     width: int = 600,
#     height: int = 400
# ) -> str:
#     """
#     Render a Google Map with a route overlay between two points.

#     Args:
#         origin_lat (float): Latitude of the origin point.
#         origin_lng (float): Longitude of the origin point.
#         dest_lat (float): Latitude of the destination point.
#         dest_lng (float): Longitude of the destination point.
#         width (int, optional): Width of the map image in pixels. Default is 600.
#         height (int, optional): Height of the map image in pixels. Default is 400.

#     Returns:
#         str: HTML <img> tag with the static map, or error message.
#     """
#     # Get the encoded polyline for the route
#     url = "https://routes.googleapis.com/directions/v2:computeRoutes"
#     headers = {
#         "Content-Type": "application/json",
#         "X-Goog-Api-Key": GOOGLE_MAPS_API_KEY,
#         "X-Goog-FieldMask": "routes.polyline.encodedPolyline"
#     }
#     data = {
#         "origin": {"location": {"latLng": {"latitude": origin_lat, "longitude": origin_lng}}},
#         "destination": {"location": {"latLng": {"latitude": dest_lat, "longitude": dest_lng}}},
#         "travelMode": "DRIVE",
#         "routingPreference": "TRAFFIC_AWARE"
#     }
#     result = await async_post(url, headers, data)
#     if not result or "routes" not in result or not result["routes"]:
#         return "No route found or API error."
#     polyline = result["routes"][0].get("polyline", {}).get("encodedPolyline")
#     if not polyline:
#         return "No polyline found for route."
#     # Build the Static Maps API URL
#     static_url = (
#         f"https://maps.googleapis.com/maps/api/staticmap?size={width}x{height}"
#         f"&path=enc:{polyline}"
#         f"&markers=color:green|label:S|{origin_lat},{origin_lng}"
#         f"&markers=color:red|label:E|{dest_lat},{dest_lng}"
#         f"&key={GOOGLE_MAPS_API_KEY}"
#     )
#     # Return an HTML <img> tag for easy rendering
#     return f'<img src="{static_url}" width="{width}" height="{height}" alt="Route Map">'

# @mcp.tool()
# async def generate_route_map_html(
#     origin_lat: float,
#     origin_lng: float,
#     dest_lat: float,
#     dest_lng: float,
#     width: int = 800,
#     height: int = 600,
#     zoom: int = 7
# ) -> str:
#     """
#     Generate HTML for an interactive Google Map with a route between two points.

#     Args:
#         origin_lat (float): Latitude of the origin point.
#         origin_lng (float): Longitude of the origin point.
#         dest_lat (float): Latitude of the destination point.
#         dest_lng (float): Longitude of the destination point.
#         width (int, optional): Width of the map in pixels. Default is 800.
#         height (int, optional): Height of the map in pixels. Default is 600.
#         zoom (int, optional): Initial zoom level. Default is 7.

#     Returns:
#         str: HTML string for the interactive map.
#     """
#     api_key = GOOGLE_MAPS_API_KEY
#     html = f"""
#     <!DOCTYPE html>
#     <html>
#       <head>
#         <title>Interactive Route Map</title>
#         <meta charset="utf-8" />
#         <style>
#           #map {{
#             height: {height}px;
#             width: {width}px;
#           }}
#         </style>
#         <script src="https://maps.googleapis.com/maps/api/js?key={api_key}"></script>
#         <script>
#           function initMap() {{
#             const origin = {{ lat: {origin_lat}, lng: {origin_lng} }};
#             const destination = {{ lat: {dest_lat}, lng: {dest_lng} }};
#             const map = new google.maps.Map(document.getElementById("map"), {{
#               zoom: {zoom},
#               center: origin,
#             }});
#             const directionsService = new google.maps.DirectionsService();
#             const directionsRenderer = new google.maps.DirectionsRenderer({{
#               draggable: true,
#               map: map,
#             }});
#             directionsService.route(
#               {{
#                 origin: origin,
#                 destination: destination,
#                 travelMode: google.maps.TravelMode.DRIVING,
#               }},
#               (response, status) => {{
#                 if (status === "OK") {{
#                   directionsRenderer.setDirections(response);
#                 }} else {{
#                   window.alert("Directions request failed due to " + status);
#                 }}
#               }}
#             );
#           }}
#         </script>
#       </head>
#       <body onload="initMap()">
#         <h2>Interactive Route Map</h2>
#         <div id="map"></div>
#       </body>
#     </html>"""
#     return html

@mcp.tool()
async def generate_route_javascript(
    origin_lat: float,
    origin_lng: float,
    dest_lat: float,
    dest_lng: float,
    zoom: int = 7
) -> str:
    """
    Generate JavaScript code for the Google Maps JavaScript API to display a route from start to destination.

    Args:
        origin_lat (float): Latitude of the origin point.
        origin_lng (float): Longitude of the origin point.
        dest_lat (float): Latitude of the destination point.
        dest_lng (float): Longitude of the destination point.
        zoom (int, optional): Initial zoom level for the map. Default is 7.

    Returns:
        str: JavaScript code as a string.
    """
    api_key = GOOGLE_MAPS_API_KEY
    js_code = f"""
    <script src="https://maps.googleapis.com/maps/api/js?key={api_key}"></script>
    <script>
      function initMap() {{
        const origin = {{ lat: {origin_lat}, lng: {origin_lng} }};
        const destination = {{ lat: {dest_lat}, lng: {dest_lng} }};
        const map = new google.maps.Map(document.getElementById("map"), {{
          zoom: {zoom},
          center: origin,
        }});
        const directionsService = new google.maps.DirectionsService();
        const directionsRenderer = new google.maps.DirectionsRenderer({{
          draggable: true,
          map: map,
        }});
        directionsService.route(
          {{
            origin: origin,
            destination: destination,
            travelMode: google.maps.TravelMode.DRIVING,
          }},
          (response, status) => {{
            if (status === "OK") {{
              directionsRenderer.setDirections(response);
            }} else {{
              window.alert("Directions request failed due to " + status);
            }}
          }}
        );
      }}
      window.onload = initMap;
    </script>
    <div id="map" style="width: 800px; height: 600px;"></div>
    """
    return js_code



if __name__ == "__main__":
    mcp.run(transport='stdio')
