from __future__ import annotations

from starlette.responses import JSONResponse


class UploadSizeLimitMiddleware:
    """Bound prediction request bodies before multipart parsing can spool them."""

    def __init__(self, app, max_body_bytes: int, path: str = "/predictions"):
        self.app = app
        self.max_body_bytes = max_body_bytes
        self.path = path

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or scope["method"] != "POST" or scope["path"] != self.path:
            await self.app(scope, receive, send)
            return

        headers = {key.lower(): value for key, value in scope.get("headers", [])}
        content_length = headers.get(b"content-length")
        if content_length:
            try:
                if int(content_length) > self.max_body_bytes:
                    await JSONResponse(
                        {"detail": "Upload request is too large."}, status_code=413
                    )(scope, receive, send)
                    return
            except ValueError:
                pass

        received_bytes = 0
        response_started = False

        async def limited_receive():
            nonlocal received_bytes
            message = await receive()
            if message["type"] == "http.request":
                received_bytes += len(message.get("body", b""))
                if received_bytes > self.max_body_bytes:
                    raise _RequestBodyTooLarge
            return message

        async def track_response(message):
            nonlocal response_started
            if message["type"] == "http.response.start":
                response_started = True
            await send(message)

        try:
            await self.app(scope, limited_receive, track_response)
        except _RequestBodyTooLarge:
            if not response_started:
                await JSONResponse(
                    {"detail": "Upload request is too large."}, status_code=413
                )(scope, receive, send)


class _RequestBodyTooLarge(Exception):
    pass
