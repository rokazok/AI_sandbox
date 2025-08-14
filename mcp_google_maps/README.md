# Notes


In Google Cloud Console [Credentials](https://console.cloud.google.com/apis/credentials) dashboard, check for an API key with access to:
1. Places API (New) - to look up information on places.
2. Routes API - to get a polyline of a route from origin to destination.
3. Weather API
4. Maps Javascript API - to render an interactive route map.
5. Maps Static API - to render an image of a route.

Copy/paste the API key into the .env file (make sure the file is not tracked in git.).

# APIs

The Places API (New) charges based on data retrieved, so you specify which fields you want to return with the [FieldMask](https://developers.google.com/maps/documentation/places/web-service/place-details#fieldmask) in the header request. 


# MCP
The [Google Maps MCP Server](https://github.com/modelcontextprotocol/servers-archived/tree/HEAD/src/google-maps) is archived.

The [quickstart MCP server](https://modelcontextprotocol.io/quickstart/server) provides examples of tools, or functions the LLM can call, and demos the examples with Claude Desktop.

# Setup
Edit the Claude configuration to run the MCP server. ex.
```bash
code ~/Library/Application\ Support/Claude/claude_desktop_config.json
```
to launch VSCode to edit the configuration so that it looks like:
```json
{
  "mcpServers": {
    "maps": {
      "command": "/Users/<USER>/.local/bin/uv",
      "args": [
        "--directory",
        "/PATH/TO/REP/mcp_google_maps",
        "run",
        "google_maps.py"
      ]
    },
  }
}
```
Note: this configuration assumes we are using uv to manage the workspace. Claude Desktop will need to be restarted



# Demo
Dialog prompts:
```
Where are some good bars and restaurants for people visiting Seattle's Fremont neighborhood?

Based on tonight's weather, what are the best options?

Give me directions from downtown Seattle to Local Tide.

Give me a map of those directions.
```