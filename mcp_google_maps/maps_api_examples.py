import requests
from dotenv import load_dotenv

load_dotenv() or load_dotenv('mcp_google_maps/.env')
API_KEY = os.getenv("GOOGLE_MAPS_API_KEY")

# Coordinates for the search location (e.g., San Francisco)
center_latitude = 37.7937
center_longitude = -122.3965
radius = 500  # in meters

# Define the endpoint URL
url = "https://places.googleapis.com/v1/places:searchNearby"

# Set up the request body and headers
headers = {
    "Content-Type": "application/json",
    "X-Goog-Api-Key": API_KEY,
    "X-Goog-FieldMask": "places.displayName,places.rating,places.priceLevel,places.editorialSummary,places.types"
}

data = {
    "includedTypes": ["restaurant"],
    "maxResultCount": 10,
    "locationRestriction": {
        "circle": {
            "center": {
                "latitude": center_latitude,
                "longitude": center_longitude
            },
            "radius": radius
        }
    }
}

# Make the POST request
response = requests.post(url, headers=headers, json=data)

# Check if the request was successful
if response.status_code == 200:
    results = response.json()
    for place in results.get("places", []):
        name = place.get("displayName", {}).get("text", "N/A")
        rating = place.get("rating", "N/A")
        price_level = place.get("priceLevel", "N/A")
        summary = place.get("editorialSummary", {}).get("text", "N/A")
        print(f"Name: {name}, Rating: {rating}, Price Level: {price_level}")
        print(f"Summary: {summary}\n")
else:
    print(f"Error: {response.status_code}, {response.text}")



### ROUTES
import requests

API_KEY = "YOUR_API_KEY"

# Define the start and end points
origin = {
    "latitude": 37.419734,
    "longitude": -122.0827784
}
destination = {
    "latitude": 37.417670,
    "longitude": -122.079595
}

# Define the endpoint URL
url = "https://routes.googleapis.com/directions/v2:computeRoutes"

# Set up the request body and headers
headers = {
    "Content-Type": "application/json",
    "X-Goog-Api-Key": API_KEY,
    "X-Goog-FieldMask": "routes.duration,routes.distanceMeters,routes.polyline.encodedPolyline"
}

data = {
    "origin": {
        "location": {
            "latLng": {
                "latitude": origin["latitude"],
                "longitude": origin["longitude"]
            }
        }
    },
    "destination": {
        "location": {
            "latLng": {
                "latitude": destination["latitude"],
                "longitude": destination["longitude"]
            }
        }
    },
    "travelMode": "DRIVE",
    "routingPreference": "TRAFFIC_AWARE"
}

# Make the POST request
response = requests.post(url, headers=headers, json=data)

# Check if the request was successful
if response.status_code == 200:
    route_data = response.json()
    # Extract the first route from the response
    route = route_data.get("routes", [])[0]
    duration = route.get("duration")
    distance_meters = route.get("distanceMeters")
    
    print(f"Route Duration: {duration}")
    print(f"Distance: {distance_meters} meters")

else:
    print(f"Error: {response.status_code}, {response.text}")




### WEATHER
import requests

API_KEY = "YOUR_API_KEY"

# Location coordinates (e.g., San Francisco)
location_lat = 37.7937
location_lng = -122.3965

# Seattle
location_lat = 47.6061
location_lng = -122.3328

# Define the endpoint URL with query parameters
url = f"https://weather.googleapis.com/v1/currentConditions:get?location={location_lat},{location_lng}&key={API_KEY}"
url = f"https://weather.googleapis.com/v1/forecast/days:lookup?key={API_KEY}&location.latitude={location_lat}&location.longitude={location_lng}&days=10&pageSize=10"
url = f"https://weather.googleapis.com/v1/forecast/hours:lookup?key={API_KEY}&location.latitude={location_lat}&location.longitude={location_lng}&hours=240&pageSize=240"

# Make the GET request
response = requests.get(url)

# Check if the request was successful
if response.status_code == 200:
    weather_data = response.json()
    
    # Extract relevant weather information
    temperature_celsius = weather_data.get("temperature", {}).get("celsius")
    condition_code = weather_data.get("conditionCode")
    
    print(f"Current Temperature: {temperature_celsius}°C")
    print(f"Weather Condition Code: {condition_code}")
    
    # Example logic for recommending activities
    if temperature_celsius is not None and temperature_celsius > 20:
        if "clear" in condition_code.lower():
            print("The weather is nice! Consider a walk in the park or a trip to the beach.")
        else:
            print("It's warm but the conditions might not be ideal for outdoor activities.")
    else:
        print("It's a bit chilly or the conditions are not great. Maybe visit a museum or a cafe indoors.")

else:
    print(f"Error: {response.status_code}, {response.text}")


### ROUTE
def render_route_map(
    origin_lat,
    origin_lng,
    dest_lat,
    dest_lng,
    width=600,
    height=400
):
    """
    Render a Google Map with a route overlay between two points.

    Args:
        origin_lat (float): Latitude of the origin point.
        origin_lng (float): Longitude of the origin point.
        dest_lat (float): Latitude of the destination point.
        dest_lng (float): Longitude of the destination point.
        width (int, optional): Width of the map image in pixels. Default is 600.
        height (int, optional): Height of the map image in pixels. Default is 400.

    Returns:
        str: HTML <img> tag with the static map, or error message.
    """
    # Get the encoded polyline for the route
    url = "https://routes.googleapis.com/directions/v2:computeRoutes"
    headers = {
        "Content-Type": "application/json",
        "X-Goog-Api-Key": API_KEY,
        "X-Goog-FieldMask": "routes.polyline.encodedPolyline"
    }
    data = {
        "origin": {"location": {"latLng": {"latitude": origin_lat, "longitude": origin_lng}}},
        "destination": {"location": {"latLng": {"latitude": dest_lat, "longitude": dest_lng}}},
        "travelMode": "DRIVE",
        "routingPreference": "TRAFFIC_AWARE"
    }
    response = requests.post(url, headers=headers, json=data)
    if response.status_code != 200:
        return f"Error: {response.status_code}, {response.text}"
    result = response.json()
    if "routes" not in result or not result["routes"]:
        return "No route found or API error."
    polyline = result["routes"][0].get("polyline", {}).get("encodedPolyline")
    if not polyline:
        return "No polyline found for route."
    # Build the Static Maps API URL
    static_url = (
        f"https://maps.googleapis.com/maps/api/staticmap?size={width}x{height}"
        f"&path=enc:{polyline}"
        f"&markers=color:green|label:S|{origin_lat},{origin_lng}"
        f"&markers=color:red|label:E|{dest_lat},{dest_lng}"
        f"&key={API_KEY}"
    )
    # Return an HTML <img> tag for easy rendering
    return f'<img src="{static_url}" width="{width}" height="{height}" alt="Route Map">'

# Seattle to Portland
#route = render_route_map(origin_lat=round(47.6062, 1), origin_lng=round(-122.3321, 1), dest_lat=round(45.5152, 1), dest_lng=round(-122.6722, 1))


def generate_interactive_route_map_html(
    origin_lat,
    origin_lng,
    dest_lat,
    dest_lng,
    api_key,
    width=800,
    height=600,
    zoom=7
):
    """
    Generate HTML for an interactive Google Map with a route between two points.

    Args:
        origin_lat (float): Latitude of the origin point.
        origin_lng (float): Longitude of the origin point.
        dest_lat (float): Latitude of the destination point.
        dest_lng (float): Longitude of the destination point.
        api_key (str): Google Maps JavaScript API key.
        width (int, optional): Width of the map in pixels. Default is 800.
        height (int, optional): Height of the map in pixels. Default is 600.
        zoom (int, optional): Initial zoom level. Default is 7.

    Returns:
        str: HTML string for the interactive map.
    """
    html = f"""
    <!DOCTYPE html>
    <html>
      <head>
        <title>Interactive Route Map</title>
        <meta charset="utf-8" />
        <style>
          #map {{
            height: {height}px;
            width: {width}px;
          }}
        </style>
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
        </script>
      </head>
      <body onload="initMap()">
        <h2>Interactive Route Map</h2>
        <div id="map"></div>
      </body>
    </html>
    """
    return html

# route = generate_interactive_route_map_html(origin_lat=round(47.6062, 1), 
#             origin_lng=round(-122.3321, 1), 
#             dest_lat=round(45.5152, 1), 
#             dest_lng=round(-122.6722, 1),
#             api_key=API_KEY
#         )

# from IPython.display import HTML
# HTML(route)

