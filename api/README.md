# API

Purpose:  
1. Create an API with FastAPI.
2. Demonstrate different functionalities of FastAPI.

This module will set up an API to tell joke and limericks, quote famous authors, and add numbers. It is set up in a silly way to show different routes fulfilling different tasks.

Run this in terminal with:
```bash
fastapi dev api/jokes_api.py
```
This will start your dev deployment server and documentation at
```
http://127.0.0.1:8000
http://127.0.0.1:8000/docs
```
where you can try test API calls.



## Explanation
A user calls an API with a request and the API gives a response. The request [consists](https://blog.postman.com/what-are-the-components-of-an-api/) of:
1. Endpoint
2. Method
3. Parameters
4. Request headers
5. Request body  

These components can be set with [FastAPI](https://fastapi.tiangolo.com/), a library for quick-and-easy API deployment that's less obtuse than Flask. The magic is in the function decorator. FastAPI handles almost everything else.

Helpful things to know:
1. FastAPI lets you define the HTTP methods (GET, POST, PUT, DELETE) to do anything. They are not constrained to those actions.
   1. If you want to check your API call in the browser, it should be a GET method.
2. Use type hints.
   1. You can see this in action with the math() command. If you remove the type hints for the output, you can get errors with multiply, ex. ValueError: [TypeError("'numpy.int64' object is not iterable"), TypeError('vars() argument must have __dict__ attribute')]
3. Default to None for optional query parameters.


## Structure
Here is the structure of this API
```
|- joke/
|  |- POST to get a joke with `type` endpoint and `about` as an optional query parameter.
|- limerick/
|  |- POST to get a limerick with `about` as an optional query parameter.
|- author/
|  |- GET to get a quote from a famous author with `author` as an optional query parameter.
|- math/  (all are PUT methods, with any number of `n` query parameters)
   |- add/ 
   |- subtract/
   |- multiply
```

These methods are purposely inconsistent to show how you can customize the API calls.
The routes demonstrate different functionalities 
* `joke` 
  * predefined routes: `knock-knock`, `chicken`, and `generic`.
  * Passing query parameters to the function, including a topic parameter.
* `limerick` or `author`
  * Passing query parameters to the function, including a topic or author parameter.
* `math`
  * Passing the route into the function to decide what kind of math operation to perform.
  * Allowing an open-ended number of `n` values.