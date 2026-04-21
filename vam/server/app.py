from fastapi import FastAPI

from vam.config import get_settings
from vam.server.api.frames import router as frames_router
from vam.server.ws.agent_chat import router as agent_chat_router

def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(title=settings.app_name, version=settings.app_version)

    # Optional API surfaces for the library-first runtime.
    app.include_router(frames_router)
    app.include_router(agent_chat_router)

    @app.get("/", include_in_schema=False)
    async def root_status():
        return {
            "name": settings.app_name,
            "version": settings.app_version,
            "mode": "library-with-optional-api",
            "docs": "/docs",
            "websocket": settings.websocket_path,
            "tui": "vam-tui",
        }

    return app


app = create_app()
