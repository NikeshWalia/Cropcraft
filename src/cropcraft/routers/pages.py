"""HTML page routes."""

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse

from cropcraft.services import fertilizer as fertilizer_service
from cropcraft.templating import templates

router = APIRouter(tags=["pages"])


@router.get("/", response_class=HTMLResponse, name="index")
async def index(request: Request) -> HTMLResponse:
    """Landing page."""
    return templates.TemplateResponse(request, "index.html", {"active": "home"})


@router.get("/crop", response_class=HTMLResponse, name="crop_form")
async def crop_form(request: Request) -> HTMLResponse:
    """Crop recommendation form."""
    return templates.TemplateResponse(request, "crop_form.html", {"active": "crop"})


@router.get("/fertilizer", response_class=HTMLResponse, name="fertilizer_form")
async def fertilizer_form(request: Request) -> HTMLResponse:
    """Fertilizer form, with the crop list driven from Crop_NPK.csv."""
    return templates.TemplateResponse(
        request,
        "fertilizer_form.html",
        {"active": "fertilizer", "crops": fertilizer_service.supported_crops()},
    )


@router.get("/pesticide", response_class=HTMLResponse, name="pest_form")
async def pest_form(request: Request) -> HTMLResponse:
    """Pest image upload form."""
    return templates.TemplateResponse(request, "pest_form.html", {"active": "pesticide"})
