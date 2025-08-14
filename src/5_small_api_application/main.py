## to run run the following in command line
## PYTHONPATH="." uv run uvicorn main:app --reload

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, PlainTextResponse
from fastapi.templating import Jinja2Templates
from fastapi import Body

from collections import deque

app = FastAPI()
templates = Jinja2Templates(directory="templates")

# In-memory storage of last 5 plan objects
plans_store = deque(maxlen=5)


@app.get("/", response_class=HTMLResponse)
async def get_latest(request: Request):
    return templates.TemplateResponse("index.html", {
        "request": request,
        "plans": list(plans_store)  # Convert deque to list for template rendering
    })


@app.post("/update", response_class=PlainTextResponse)
async def update_data(plan: dict = Body(...)):
    plans_store.appendleft(plan)  # Add to left, so latest is shown first
    return "✅ Plan added successfully."