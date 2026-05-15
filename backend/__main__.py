"""Entry point: uv run python -m backend"""

import uvicorn

from config.settings import settings

if __name__ == "__main__":
    uvicorn.run(
        "backend.api.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
    )
