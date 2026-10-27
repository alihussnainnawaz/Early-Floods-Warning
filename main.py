"""Start here. This file creates the web app and connects the pieces.

Run it with:   uvicorn main:app --reload
Then open:     http://127.0.0.1:8000

Where things are:
  backend/routes.py        the API endpoints (/api/forecast, /api/flow/..., ...)
  backend/model_store.py   loads the trained model file
  backend/live_data.py     downloads and caches river flow
  backend/about_content.py text shown in the "how well it did" section
  backend/config.py        settings (paths, cache times)
  pipeline_lib.py          data + model code shared with the training notebook
  static/                  the web page (HTML, CSS, JavaScript)
"""
from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from backend.config import STATIC_DIR
from backend.routes import router

app = FastAPI(title="Sindh river flood early warning (research prototype)")

app.include_router(router)                                          # the /api/... endpoints
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")  # css, js


@app.get("/", include_in_schema=False)
def index():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/flow", include_in_schema=False)
def flow_page():
    return FileResponse(STATIC_DIR / "flow.html")
