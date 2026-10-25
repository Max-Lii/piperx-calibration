from __future__ import annotations

import uvicorn

from .app import create_app
from .settings import load_settings


def main() -> None:
    settings = load_settings()
    uvicorn.run(create_app(settings=settings), host=settings.host, port=settings.port, workers=1)


if __name__ == "__main__":
    main()
