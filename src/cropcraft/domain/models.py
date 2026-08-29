"""Immutable domain types shared by the services and the templates."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class NutrientAdvice:
    """Advice for one soil nutrient being high, low or on target."""

    headline: str
    tips: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class Pesticide:
    """A registered product and its label dose, as shown in CropCraft 1.x."""

    name: str
    dose: str
    image: str


@dataclass(frozen=True, slots=True)
class Pest:
    """A pest the classifier can identify, plus how to deal with it."""

    slug: str
    name: str
    description: str
    damage: str
    pesticides: tuple[Pesticide, ...] = ()
    cultural_controls: tuple[str, ...] = ()
    beneficial: bool = False
