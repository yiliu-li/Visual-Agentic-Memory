import os
import asyncio
from pathlib import Path

def run_tui():
    """Entry point for the terminal TUI."""
    from vam.tui import main
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nInterrupted.")

def run_server():
    """Entry point for the API server."""
    import uvicorn

    port = int(os.environ.get("PORT", 8000))
    is_prod = "RENDER" in os.environ or "PORT" in os.environ
    reload_env = os.environ.get("UVICORN_RELOAD")
    if reload_env is not None:
        reload = reload_env.strip().lower() in ("1", "true", "yes", "on")
    else:
        reload = (not is_prod) and (os.name != "nt")
    
    package_root = Path(__file__).resolve().parent
    print(f"Starting VAM server on port {port} (reload={reload})...")
    uvicorn.run(
        "vam.server.app:app", 
        host="0.0.0.0", 
        port=port, 
        log_level="info" if is_prod else "debug", 
        reload=reload, 
        reload_dirs=[str(package_root)] if reload else None,
        ws_max_size=1024 * 1024 * 1024 * 2
    )
