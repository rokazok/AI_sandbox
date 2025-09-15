import os
from dotenv import load_dotenv
import requests

def parse_user_query(query):
    # TODO: Use NLP or regex to extract location, cuisine, seating, date/time
    # For now, return a dummy context
    return {
        "location": "Ballard Locks",
        "cuisine": "seafood",
        "outdoor_seating": True,
        "datetime": "tomorrow night"
    }

def get_weather_forecast(context, api_key, mcp_url):
    location = context['location']
    datetime = context['datetime']
    url = f"{mcp_url}/weather?location={location}&datetime={datetime}&key={api_key}"
    try:
        response = requests.get(url)
        if response.status_code == 200:
            return response.json()
        else:
            return {'forecast': 'unknown', 'temperature': 'unknown', 'error': response.text}
    except Exception:
        return {'forecast': 'unknown', 'temperature': 'unknown', 'error': 'Request failed'}

def search_restaurants(context, api_key, mcp_url):
    location = context['location']
    cuisine = context['cuisine']
    url = f"{mcp_url}/places?location={location}&type=restaurant&cuisine={cuisine}&key={api_key}"
    try:
        response = requests.get(url)
        if response.status_code == 200:
            restaurants = response.json().get('results', [])
            filtered = [r for r in restaurants if cuisine.lower() in r.get('cuisine', '').lower()]
            sorted_restaurants = sorted(filtered, key=lambda x: x.get('rating', 0), reverse=True)
            return sorted_restaurants[:5]
        else:
            return []
    except Exception:
        return []

def get_outdoor_seating_advice(weather, context):
    if context.get('outdoor_seating') and weather.get('forecast') == 'rain':
        return "Rain is predicted. Consider indoor seating."
    elif context.get('outdoor_seating'):
        return "Outdoor seating should be fine!"
    else:
        return "No outdoor seating requested."

def main():
    load_dotenv()
    google_maps_api_key = os.getenv("GOOGLE_MAPS_API_KEY")
    mcp_server_url = os.getenv("MCP_SERVER_URL")

    print("Welcome to the Restaurant Finder App!")
    user_query = input("Describe your ideal restaurant experience (e.g., 'What's a good seafood restaurant near the Ballard Locks with nice outdoor dining for tomorrow night?'): ")

    # Step 1: Parse user query for context
    context = parse_user_query(user_query)
    print(f"Parsed context: {context}")

    # Step 2: Get weather forecast for requested location/time
    weather = get_weather_forecast(context, google_maps_api_key, mcp_server_url)
    print(f"Weather forecast: {weather}")

    # Step 3: Search for restaurants near location, filter by cuisine/rating
    restaurants = search_restaurants(context, google_maps_api_key, mcp_server_url)

    # Step 4: Advise on outdoor seating if rain predicted
    advice = get_outdoor_seating_advice(weather, context)

    # Step 5: Display results
    print("\nRecommended Restaurants:")
    for r in restaurants:
        print(f"- {r['name']} (Rating: {r['rating']}, Cuisine: {r['cuisine']})")
    print(f"\nAdvice: {advice}")

if __name__ == "__main__":
    main()
