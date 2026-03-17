"""
main.py — Estadística Core · FastAPI
"""
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from routers import upload, descriptiva, inferencia, visualizaciones

app = FastAPI(title="Estadística Core · TL", version="1.0.0")

# Routers
app.include_router(upload.router,           prefix="/api")
app.include_router(descriptiva.router,      prefix="/api")
app.include_router(inferencia.router,       prefix="/api")
app.include_router(visualizaciones.router,  prefix="/api")

# Static files (frontend)
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
def root():
    return FileResponse("static/index.html")
