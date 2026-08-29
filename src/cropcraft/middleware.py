"""Security response headers.

CropCraft 1.x set none of these, and served an inline-script page over a
Werkzeug debugger. Every script and style is now a same-origin file, so the
policy below can forbid inline execution outright.
"""

from collections.abc import Awaitable, Callable

from fastapi import Request, Response

CONTENT_SECURITY_POLICY = "; ".join(
    (
        "default-src 'self'",
        "script-src 'self'",
        "style-src 'self'",
        "img-src 'self' data: blob:",  # blob: for the local upload preview
        "font-src 'self'",
        "connect-src 'self'",
        "form-action 'self'",
        "frame-ancestors 'none'",
        "base-uri 'self'",
        "object-src 'none'",
    )
)

# Swagger UI and ReDoc are served by FastAPI from a CDN and bootstrap themselves
# with an inline script, so the policy above would blank them out. They get a
# scoped exception rather than the whole application being loosened for them.
DOCS_CONTENT_SECURITY_POLICY = "; ".join(
    (
        "default-src 'self'",
        "script-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net",
        "style-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net",
        "img-src 'self' data: https://fastapi.tiangolo.com",
        "font-src 'self' https://cdn.jsdelivr.net",
        "connect-src 'self'",
        "frame-ancestors 'none'",
        "base-uri 'self'",
        "object-src 'none'",
    )
)

DOCS_PATHS = ("/docs", "/redoc")

SECURITY_HEADERS = {
    "Content-Security-Policy": CONTENT_SECURITY_POLICY,
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
    "Referrer-Policy": "strict-origin-when-cross-origin",
    "Permissions-Policy": "geolocation=(), microphone=(), camera=()",
    "Cross-Origin-Opener-Policy": "same-origin",
}


async def security_headers(
    request: Request, call_next: Callable[[Request], Awaitable[Response]]
) -> Response:
    """Attach the security headers to every response."""
    response = await call_next(request)
    for header, value in SECURITY_HEADERS.items():
        response.headers.setdefault(header, value)

    if request.url.path.startswith(DOCS_PATHS):
        response.headers["Content-Security-Policy"] = DOCS_CONTENT_SECURITY_POLICY

    return response
