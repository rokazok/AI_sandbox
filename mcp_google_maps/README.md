# Model Context Protocol
This section tests Model Context Protocol (MCP) with Claude and the Google Maps API, where Claude decides from
user prompts whether to hit Google Maps for restaurant reviews, weather, and driving directions.

## Setup
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
        "/PATH/TO/REPO/mcp_google_maps",
        "run",
        "google_maps.py"
      ],
      "env": {
        "GOOGLE_MAPS_API_KEY": "YOUR_API_KEY_HERE"
      }
    },
  }
}
```
Note: this configuration assumes we are using uv to manage the workspace on MacOS. Claude Desktop will need to be restarted

Once configured, you should be able to see the available tools in Claude's chat window settings icon or in Claude Settings > Developer.

## Explanation
`google_maps.py` contains the Google Maps API calls. It loads the `mcp` [library](https://github.com/modelcontextprotocol/python-sdk), which handles the integrations with Claude. 
The `mcp.tool()` decorator registers the functions as tools that Claude can use.


## Notes

# APIs
In Google Cloud Console [Credentials](https://console.cloud.google.com/apis/credentials) dashboard, check for an API key with access to:
1. Places API (New) - to look up information on places.
2. Routes API - to get a polyline of a route from origin to destination.
3. Weather API
4. Maps Javascript API - to render an interactive route map.
5. Maps Static API - to render an image of a route.

Copy/paste the API key into the .env file (make sure the file is not tracked in git.).

Notes:
The Places API (New) charges based on data retrieved, so you specify which fields you want to return with the [FieldMask](https://developers.google.com/maps/documentation/places/web-service/place-details#fieldmask) in the header request. 


## MCP
The [Google Maps MCP Server](https://github.com/modelcontextprotocol/servers-archived/tree/HEAD/src/google-maps) is archived, so we will create API calls here.

The [quickstart MCP server](https://modelcontextprotocol.io/quickstart/server) provides examples of tools, or functions the LLM can call, and demos the examples with Claude Desktop.





# Demo
1. Start Claude for desktop.
  1. Clicking on the Search and Tools button should expand the menu and allow you to toggle your MCP tools on/off if they are available. Make sure they're available and on.
2. Prompt Claude, and it should use the MCP tools as needed.

Dialog prompts:
```
Where are some good bars and restaurants for people visiting Seattle's Fremont neighborhood?

Based on tonight's weather, what are the best options?

Give me directions from downtown Seattle to Local Tide.

Give me a map of those directions.
```