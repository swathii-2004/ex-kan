from fastapi import FastAPI

from app.database import get_db

app = FastAPI(title="Ex-KAN API")


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
