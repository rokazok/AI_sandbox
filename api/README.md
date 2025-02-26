# API

Purpose:  
1. Create an API with FastAPI

Run this in terminal with:
```bash
fastapi dev api/jokes_api.py
```

This module will set up an API to:
1. tell jokes
2. tell limericks
3. quote famous authors
4. add numbers
This is a silly example to show different routes fulfilling different tasks.

## Explanation
[FastAPI](https://fastapi.tiangolo.com/) is a quick-and-easy way to deploy an API, and it's less obtuse than Flask.

A user calls an API with a request and the API gives a response. The request [consists](https://blog.postman.com/what-are-the-components-of-an-api/) of:
1. Endpoint
2. Method
3. Parameters
4. Request headers
5. Request body  

These components can be set with FastAPI.

Helpful things to know:
1. You can define the HTTP methods (GET, POST, PUT, DELETE) to do anything. They are not constrained to those actions.
   1. If you want to check your API call in the browser, it should be a GET method.
2. Use type hints.
   1. You can see this in action with the math() command. If you remove the type hints for the output, you can get errors with multiply, ex. ValueError: [TypeError("'numpy.int64' object is not iterable"), TypeError('vars() argument must have __dict__ attribute')]
3. Default to None for optional query parameters.


## Structure
Here is the structure of this API
```
|- joke/
|  |- POST to get a joke
|- limerick/
|  |- POST to get a limerick
|- author/
|  |- GET to get a quote from a famous author
|- math/  (all are PUT commands)
   |- add/ 
   |- subtract/
   |- multiply
```

These methods are purposely inconsistent to show how you can customize the API calls.
The routes demonstrate different functionalities 
* `joke` 
  * predefined routes: `knock-knock`, `chicken`, and `None`.
  * Passing query parameters to the function, including a topic parameter.
* `limerick` or `author`
  * Passing query parameters to the function, including a topic or author parameter.
* `math`
  * Passing the route into the function to decide what kind of math operation to perform.
  * Allowing an open-ended number of `n` values.