
import uvicorn
import sys
import os
from pathlib import Path

if __name__ == "__main__":
    try:
        # Production (Render) uses PORT env var
        port = int(os.environ.get("PORT", 8000))
        # Disable reload in production and on Windows unless explicitly enabled.
        is_prod = "RENDER" in os.environ or "PORT" in os.environ
        reload_env = os.environ.get("UVICORN_RELOAD")
        if reload_env is not None:
            reload = reload_env.strip().lower() in ("1", "true", "yes", "on")
        else:
            reload = (not is_prod) and (os.name != "nt")
        package_root = Path(__file__).resolve().parent / "vam"

        print(f"Starting uvicorn on port {port} (reload={reload})...")
        uvicorn.run(
            "vam.server.app:app", 
            host="0.0.0.0", 
            port=port, 
            log_level="info" if is_prod else "debug", 
            reload=reload, 
            reload_dirs=[str(package_root)] if reload else None,
            ws_max_size=1024 * 1024 * 1024 * 2  # 2GB WebSocket limit
        )
    except Exception as e:
        print(f"Error starting uvicorn: {e}")
        sys.exit(1)
