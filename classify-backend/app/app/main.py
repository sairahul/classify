from fastapi import FastAPI
from starlette.middleware.cors import CORSMiddleware
from starlette.requests import Request
from starlette.staticfiles import StaticFiles

from app.api.api_v1.api import api_router
from app.core import config
from app.db.session import Session

app = FastAPI(title=config.PROJECT_NAME, openapi_url="/api/v1/openapi.json")

# CORS
origins = []

# Set all CORS enabled origins
if config.BACKEND_CORS_ORIGINS:
    origins_raw = config.BACKEND_CORS_ORIGINS.split(",")
    for origin in origins_raw:
        use_origin = origin.strip()
        origins.append(use_origin)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    ),

app.include_router(api_router, prefix=config.API_V1_STR)


@app.middleware("http")
async def db_session_middleware(request: Request, call_next):
    request.state.db = Session()
    response = await call_next(request)
    request.state.db.close()
    return response


def add_static_routes():
    from app.db_models.dataset import Dataset as DBDataset

    db = Session()
    datasets = db.query(DBDataset).all()
    for dataset in datasets:
        app.mount(config.DATASET_STATIC_ORIG_TEMPLATE.format(dataset_id=dataset.id),
                  StaticFiles(directory=dataset.base_dir),
                  name="static")
        app.mount(config.DATASET_STATIC_THUMB_TEMPLATE.format(dataset_id=dataset.id),
                  StaticFiles(directory=dataset.thumbnail_dir),
                  name="static")
    db.close()

add_static_routes()
