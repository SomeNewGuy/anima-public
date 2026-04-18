"""Plugin endpoints — FastAPI APIRouter.

Wire this in plugin.py register_endpoints():

    from plugins.my_plugin.endpoints import router as ep_router
    ep_router._get_engine = get_engine
    app.include_router(ep_router)
"""

import logging
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

logger = logging.getLogger("plugins.my-plugin.endpoints")

router = APIRouter()

# Set by plugin.register_endpoints()
_get_engine = None


@router.get("/my-plugin/status")
async def plugin_status():
    """Plugin status endpoint."""
    cli = _get_engine()
    if not cli:
        return JSONResponse({"error": "Plugin not initialized"}, status_code=400)

    try:
        db = cli.evolution.semantic.db_conn
        beliefs = db.execute(
            "SELECT COUNT(*) FROM beliefs WHERE COALESCE(deprecated,0)=0"
        ).fetchone()[0]
        edges = db.execute(
            "SELECT COUNT(*) FROM belief_links WHERE COALESCE(active,1)=1"
        ).fetchone()[0]

        return {
            "graph": {
                "beliefs": beliefs,
                "edges": edges,
                "integration_pct": round(edges / max(beliefs, 1) * 100, 1),
            },
        }
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)
