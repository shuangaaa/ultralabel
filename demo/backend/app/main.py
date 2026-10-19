from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from ..app_conf import settings
from .routes import router


def create_app(data_dir=None):
    app = FastAPI(title="UltraLabel API", version="0.1.0")
    app.state.data_dir = data_dir or settings.data_dir
    app.state.inference_backend = None
    app.add_middleware(CORSMiddleware,
                       allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
                       allow_credentials=False, allow_methods=["*"], allow_headers=["*"])
    app.include_router(router)
    return app


app = create_app()
