"""Shared Jinja environment.

Autoescaping is on for every template. CropCraft 1.x wrapped its advice text in
``Markup()``, which turned escaping off for that content.
"""

from fastapi.templating import Jinja2Templates

from cropcraft.config import PROJECT_ROOT

templates = Jinja2Templates(directory=str(PROJECT_ROOT / "templates"))


def _percent(value: float) -> str:
    """Render a 0-1 probability as a percentage."""
    return f"{value * 100:.1f}%"


templates.env.filters["percent"] = _percent
