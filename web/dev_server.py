"""Local preview of the Vercel deployment: API + static UI on one port.

    uv run python web/dev_server.py   # http://127.0.0.1:8000

On Vercel, ``public/`` is served statically and ``/api/*`` reaches
``api/index.py`` (see ``vercel.json``); this script reproduces that with uvicorn.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

WEB_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(WEB_ROOT))

from fastapi.staticfiles import StaticFiles  # noqa: E402

from api.index import app  # noqa: E402

app.mount("/", StaticFiles(directory=WEB_ROOT / "public", html=True), name="public")

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=int(os.environ.get("PORT", "8000")))
