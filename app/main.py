"""
FastAPI application entrypoint.

Step 1 reconciles what used to be two separate app.mount("/static", ...)
calls (one implied per notebook) into a single instance here, and adds the
/outputs mount Module 2 will need once it's wired in (Step 13+).
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from . import config
from .routers import module1

app = FastAPI(title="Badminton Performance Analysis API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

if config.STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(config.STATIC_DIR)), name="static")

app.mount("/outputs", StaticFiles(directory=str(config.OUTPUTS_DIR)), name="outputs")

app.include_router(module1.router)
# app.include_router(module2.router)  # added starting Step 13


@app.get("/")
async def read_index():
    index_file = config.STATIC_DIR / "index.html"
    if index_file.exists():
        return FileResponse(str(index_file))
    return JSONResponse({"message": "Badminton analysis API is running."})
