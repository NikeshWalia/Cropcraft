"""Prediction routes.

Each recommendation is exposed twice: an HTML route for the browser forms and
a JSON route under ``/api`` for programmatic use. Both share one service call,
so the two can't drift apart.
"""

from typing import Annotated

from fastapi import (
    APIRouter,
    Depends,
    File,
    Form,
    HTTPException,
    Request,
    UploadFile,
    status,
)
from fastapi.responses import HTMLResponse

from cropcraft.config import get_settings
from cropcraft.schemas import (
    CropRequest,
    CropResponse,
    FertilizerRequest,
    FertilizerResponse,
    PestResponse,
)
from cropcraft.services import crop as crop_service
from cropcraft.services import fertilizer as fertilizer_service
from cropcraft.services import pest as pest_service
from cropcraft.templating import templates

router = APIRouter()


async def read_upload(image: Annotated[UploadFile, File()]) -> bytes:
    """Read an uploaded image, enforcing the type and size limits.

    The declared content type is checked first as a cheap filter, but it is
    attacker-controlled, so the real guarantee comes from Pillow decoding the
    bytes downstream. The body is read in chunks and abandoned the moment it
    exceeds the cap, so an oversized upload is never fully buffered.
    """
    settings = get_settings()

    if image.content_type not in settings.allowed_image_types:
        raise HTTPException(
            status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"Upload a JPEG, PNG or WebP image (got {image.content_type or 'nothing'}).",
        )

    payload = bytearray()
    while chunk := await image.read(64 * 1024):
        payload.extend(chunk)
        if len(payload) > settings.max_upload_bytes:
            limit_mb = settings.max_upload_bytes // (1024 * 1024)
            raise HTTPException(
                status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=f"Image must be smaller than {limit_mb} MB.",
            )

    if not payload:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, detail="No image was uploaded.")
    return bytes(payload)


# --------------------------------------------------------------------------
# HTML form handlers
# --------------------------------------------------------------------------

html_router = APIRouter(tags=["forms"])


@html_router.post("/crop", response_class=HTMLResponse, name="crop_predict")
async def crop_predict(request: Request, form: Annotated[CropRequest, Form()]):
    """Recommend a crop and render the result page."""
    try:
        result = crop_service.recommend(form)
    except crop_service.ModelUnavailableError as exc:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)) from exc

    return templates.TemplateResponse(
        request,
        "crop_result.html",
        {
            "active": "crop",
            "result": result,
            "readings": form,
            "npk_crop": fertilizer_service.normalise_crop(result.crop),
        },
    )


@html_router.post("/fertilizer", response_class=HTMLResponse, name="fertilizer_predict")
async def fertilizer_predict(request: Request, form: Annotated[FertilizerRequest, Form()]):
    """Compare soil NPK against the crop's needs and render the advice."""
    try:
        result = fertilizer_service.recommend(form)
    except fertilizer_service.UnknownCropError as exc:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc

    return templates.TemplateResponse(
        request, "fertilizer_result.html", {"active": "fertilizer", "result": result}
    )


@html_router.post("/pesticide", response_class=HTMLResponse, name="pest_predict")
async def pest_predict(request: Request, payload: Annotated[bytes, Depends(read_upload)]):
    """Identify the pest in an uploaded image and render the treatment page."""
    try:
        result = pest_service.identify(payload)
    except pest_service.InvalidImageError as exc:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    except pest_service.ModelUnavailableError as exc:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)) from exc

    return templates.TemplateResponse(
        request, "pest_result.html", {"active": "pesticide", "result": result}
    )


# --------------------------------------------------------------------------
# JSON API
# --------------------------------------------------------------------------

api_router = APIRouter(prefix="/api", tags=["api"])


@api_router.post("/crop", response_model=CropResponse)
async def api_crop(payload: CropRequest) -> CropResponse:
    """Recommend a crop for the given soil and weather readings."""
    try:
        return crop_service.recommend(payload)
    except crop_service.ModelUnavailableError as exc:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)) from exc


@api_router.post("/fertilizer", response_model=FertilizerResponse)
async def api_fertilizer(payload: FertilizerRequest) -> FertilizerResponse:
    """Advise on fertilizer for a crop given measured NPK."""
    try:
        return fertilizer_service.recommend(payload)
    except fertilizer_service.UnknownCropError as exc:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc


@api_router.post("/pesticide", response_model=PestResponse)
async def api_pesticide(payload: Annotated[bytes, Depends(read_upload)]) -> PestResponse:
    """Identify the pest in an uploaded image."""
    try:
        return pest_service.identify(payload)
    except pest_service.InvalidImageError as exc:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    except pest_service.ModelUnavailableError as exc:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)) from exc


@api_router.get("/crops")
async def api_crops() -> dict[str, list[str]]:
    """List the crops each subsystem knows about."""
    try:
        predictable = crop_service.crop_classes()
    except crop_service.ModelUnavailableError:
        predictable = []
    return {"predictable": predictable, "npk_reference": fertilizer_service.supported_crops()}


@api_router.get("/metrics")
async def api_metrics() -> dict[str, dict]:
    """Held-out accuracy for both models, as written by the training scripts.

    Published so the numbers can be checked against the README rather than
    taken on trust. Both are measured on data withheld from training and from
    model selection.
    """
    return {"crop": crop_service.metrics(), "pest": pest_service.metrics()}
