from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.database import get_db
from app.routers import dashboard, explain, predict

app = FastAPI(title="Ex-KAN API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(predict.router)
app.include_router(explain.router)
app.include_router(dashboard.router)


@app.get("/health")
def health_check() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/db-check")
def db_check() -> dict[str, str]:
    try:
        get_db().command("ping")
        return {"database": "connected"}
    except Exception as exc:
        return {"database": "error", "detail": str(exc)}
