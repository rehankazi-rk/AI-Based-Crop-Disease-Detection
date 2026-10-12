from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from backend.api import auth as auth_router
from backend.api import history as history_router
from backend.api import prediction as prediction_router
from backend.database.database import init_db
from backend.middleware import UploadSizeLimitMiddleware
from backend.services.image_processor import MAX_IMAGE_BYTES

@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    init_db()
    yield


app = FastAPI(
    title="Crop Disease Detection API",
    version="1.0.0",
    description=(
        "Secure AI-powered crop disease detection backend for the MVP. "
        "Users can register, log in, upload leaf images, and receive disease predictions."
    ),
    lifespan=lifespan,
)

FRONTEND_DIR = Path(__file__).resolve().parent / "frontend"
app.mount("/static", StaticFiles(directory=FRONTEND_DIR / "static"), name="static")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(
    UploadSizeLimitMiddleware,
    max_body_bytes=MAX_IMAGE_BYTES + 1024 * 1024,
)


app.include_router(auth_router.router)
app.include_router(history_router.router)
app.include_router(prediction_router.router)


@app.get("/")
def read_root():
    return FileResponse(FRONTEND_DIR / "index.html")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
