# Copyright (C) 2026 Gerald Teeple
#
# This file is part of ANIMA.
#
# ANIMA is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# ANIMA is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with ANIMA. If not, see <https://www.gnu.org/licenses/>.

"""Plugin proxy — forwards requests from core to plugin worker processes.

Workers run on localhost only (127.0.0.1). Users never hit them directly.
Core handles all user-facing HTTP, proxies to workers internally.

Supports: GET/POST/PUT/DELETE, streaming (SSE), timeouts, error handling.
"""

import httpx
from fastapi import Request, Response
from fastapi.responses import StreamingResponse

TIMEOUT = httpx.Timeout(300.0, connect=5.0)  # 5 min — LLM planning passes take time


async def forward_request(worker_url: str, path: str, request: Request) -> Response:
    """Forward an HTTP request to a plugin worker.

    Args:
        worker_url: Worker base URL (e.g. http://127.0.0.1:9002)
        path: Request path after plugin prefix
        request: Original FastAPI request

    Returns:
        Response from worker, or error response on timeout/unreachable.
    """
    url = f"{worker_url.rstrip('/')}/{path.lstrip('/')}"

    headers = dict(request.headers)
    headers.pop("host", None)

    params = dict(request.query_params)
    method = request.method
    body = await request.body()

    # Check for SSE — needs special handling (client must stay alive)
    if "text/event-stream" in headers.get("accept", ""):
        client = httpx.AsyncClient(timeout=httpx.Timeout(None))  # no timeout for SSE
        try:
            req = client.build_request(
                method, url, headers=headers, params=params, content=body,
            )
            upstream = await client.send(req, stream=True)

            async def stream_body():
                try:
                    async for chunk in upstream.aiter_raw():
                        yield chunk
                finally:
                    await upstream.aclose()
                    await client.aclose()

            return StreamingResponse(
                stream_body(),
                status_code=upstream.status_code,
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "X-Accel-Buffering": "no",
                },
            )
        except Exception:
            await client.aclose()
            raise

    # Normal (non-SSE) request
    async with httpx.AsyncClient(timeout=TIMEOUT) as client:
        try:
            resp = await client.request(
                method, url, headers=headers, params=params, content=body,
            )
            return Response(
                content=resp.content,
                status_code=resp.status_code,
                headers={
                    k: v for k, v in resp.headers.items()
                    if k.lower() not in ("content-length", "transfer-encoding")
                },
            )

        except httpx.TimeoutException:
            return Response(
                content=b'{"error":"worker_timeout"}',
                status_code=504,
                media_type="application/json",
            )

        except httpx.RequestError as e:
            return Response(
                content=f'{{"error":"worker_unreachable","detail":"{e}"}}'.encode(),
                status_code=502,
                media_type="application/json",
            )
