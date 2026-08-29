"""CropCraft application factory."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.exception_handlers import http_exception_handler
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from cropcraft.config import PROJECT_ROOT, get_settings
from cropcraft.middleware import security_headers
from cropcraft.routers import pages, predictions
from cropcraft.templating import templates

DESCRIPTION = """
Crop, fertilizer and pesticide recommendations from soil readings and pest photos.

* `POST /api/crop` -- recommend a crop from N, P, K, temperature, humidity, pH and rainfall
* `POST /api/fertilizer` -- compare measured NPK against a crop's requirement
* `POST /api/pesticide` -- identify a pest from an uploaded image
"""


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Warm the crop model at startup so the first request isn't slow.

    The pest model is left to load lazily -- importing TensorFlow costs several
    seconds, and most deployments will serve the crop and fertilizer endpoints
    far more often.
    """
    from cropcraft.services import crop

    try:
        crop.crop_classes()
    except crop.ModelUnavailableError as exc:
        app.state.startup_warning = str(exc)
    else:
        app.state.startup_warning = None
    yield


def create_app() -> FastAPI:
    """Build and configure the ASGI application."""
    settings = get_settings()

    app = FastAPI(
        title="CropCraft",
        description=DESCRIPTION,
        version="2.0.0",
        lifespan=lifespan,
        debug=settings.debug,
    )

    app.middleware("http")(security_headers)

    app.mount("/static", StaticFiles(directory=PROJECT_ROOT / "static"), name="static")
    app.include_router(pages.router)
    app.include_router(predictions.html_router, prefix="/predict")
    app.include_router(predictions.api_router)

    @app.get("/healthz", include_in_schema=False)
    async def healthz(request: Request) -> dict[str, str]:
        """Liveness probe.

        Reports ``degraded`` when a model artifact is missing, so a deployment
        that forgot to run the training scripts is visible without waiting for
        the first prediction to fail.
        """
        warning = getattr(request.app.state, "startup_warning", None)
        if warning:
            return {"status": "degraded", "detail": warning}
        return {"status": "ok"}

    @app.exception_handler(RequestValidationError)
    async def on_validation_error(request: Request, exc: RequestValidationError):
        """Render validation failures as a friendly page for form posts.

        1.x let a bad value raise ``ValueError`` inside the handler and
        returned a stack trace.
        """
        if request.url.path.startswith("/api"):
            return JSONResponse({"detail": exc.errors()}, status_code=422)
        problems = [
            f"{'.'.join(str(p) for p in err['loc'][1:]) or 'input'}: {err['msg']}"
            for err in exc.errors()
        ]
        return templates.TemplateResponse(
            request,
            "error.html",
            {"title": "Check your inputs", "problems": problems},
            status_code=422,
        )

    @app.exception_handler(HTTPException)
    async def on_http_error(request: Request, exc: HTTPException):
        """Render HTTP errors as a page for browsers, JSON for the API."""
        if request.url.path.startswith("/api"):
            return await http_exception_handler(request, exc)
        return templates.TemplateResponse(
            request,
            "error.html",
            {"title": "Something went wrong", "problems": [str(exc.detail)]},
            status_code=exc.status_code,
        )

    return app


app = create_app()
