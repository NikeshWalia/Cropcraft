"""Pest reference data: identification notes and recommended treatments.

CropCraft 1.x kept this as ten near-identical HTML templates that the view
selected by string-concatenating a predicted label onto ``".html"``. That made
the label list, the template list and the classifier's output three separate
things that had to be kept in sync by hand. Here the data lives in one place
and a single template renders it.

Product names and doses are carried over from the 1.x templates unchanged.
One behavioural change: **earthworm is marked beneficial**. The 1.x app
recommended malathion for it, but earthworms improve soil structure, aeration
and nutrient cycling -- treating them works against the project's own goal of
reducing soil degradation.
"""

import json
from pathlib import Path
from typing import Final

from cropcraft.domain.models import Pest, Pesticide

_PRODUCTS: Final[dict[str, list[dict[str, str]]]] = json.loads(
    (Path(__file__).parent / "_pest_products.json").read_text(encoding="utf-8")
)


def _products(slug: str) -> tuple[Pesticide, ...]:
    """Load the 1.x product cards for one pest."""
    return tuple(
        Pesticide(name=item["name"], dose=item["dose"], image=item["image"])
        for item in _PRODUCTS.get(slug, ())
    )


_DESCRIPTIONS: Final[dict[str, tuple[str, str, str, tuple[str, ...]]]] = {
    "aphids": (
        "Aphids",
        "Small soft-bodied sap-sucking insects, usually green or black, that cluster on "
        "new shoots and the undersides of leaves.",
        "Curled and yellowing leaves, stunted growth, and sticky honeydew that encourages "
        "sooty mould. Aphids also transmit several plant viruses.",
        (
            "Encourage ladybirds and lacewings, which prey on aphids.",
            "Spray a strong jet of water on light infestations before reaching for chemicals.",
            "Avoid over-applying nitrogen, which produces the soft growth aphids prefer.",
        ),
    ),
    "armyworm": (
        "Armyworm",
        "Caterpillars of night-flying moths that feed in large groups and move across a "
        "field together.",
        "Ragged holes in leaves and, in heavy attacks, whole seedlings cut down at the base. "
        "Damage builds very quickly once a group establishes.",
        (
            "Scout fields at dusk and early morning when larvae are active.",
            "Plough after harvest to expose pupae in the soil.",
            "Flood irrigation can drive larvae out of the crop.",
        ),
    ),
    "beetle": (
        "Beetle",
        "Hard-shelled chewing insects. Adults feed on foliage and flowers; the grubs of "
        "many species feed on roots below ground.",
        "Notched or skeletonised leaves, damaged flowers, and root injury that shows above "
        "ground as wilting.",
        (
            "Hand-pick adults early in the morning when they are sluggish.",
            "Rotate crops to break the soil-dwelling grub stage.",
        ),
    ),
    "bollworm": (
        "Bollworm",
        "Caterpillars that bore into buds, flowers and developing bolls or pods, feeding "
        "from inside the structure.",
        "Entry holes on bolls and pods, shed flowers, and hollowed seed. One of the most "
        "costly pests of cotton, pigeonpea and chickpea in India.",
        (
            "Use pheromone traps to time treatment to actual moth flights.",
            "Grow a trap crop such as marigold along field borders.",
            "Rotate insecticide groups to slow resistance, which is already widespread.",
        ),
    ),
    "earthworm": (
        "Earthworm",
        "A soil-dwelling annelid, not a crop pest. Earthworms break down organic matter and "
        "open channels that improve drainage and root growth.",
        "None. Earthworm activity raises soil organic matter, aeration and water infiltration. "
        "High earthworm counts are a sign of healthy soil.",
        (
            "No treatment is warranted -- earthworms are beneficial.",
            "Add organic matter and reduce tillage to encourage them.",
            "Avoid broad-spectrum soil insecticides, which reduce earthworm populations.",
        ),
    ),
    "grasshopper": (
        "Grasshopper",
        "Large jumping insects with strong chewing mouthparts that feed on leaves and stems, "
        "often migrating in from field margins.",
        "Irregular chewed leaf margins and, in severe outbreaks, complete defoliation. Damage "
        "usually starts at field edges.",
        (
            "Keep field bunds and margins clear of tall weeds where eggs are laid.",
            "Treat border strips first, where infestations typically begin.",
        ),
    ),
    "mites": (
        "Mites",
        "Tiny eight-legged arachnids, barely visible without a lens, that feed on the "
        "undersides of leaves. Populations explode in hot, dry weather.",
        "Fine pale stippling on leaves, bronzing, and delicate webbing. Heavy infestations "
        "cause leaves to dry and drop.",
        (
            "Mites are arachnids, not insects -- use a miticide, not a general insecticide.",
            "Broad-spectrum sprays often make mites worse by killing predatory mites.",
            "Maintain irrigation; drought-stressed crops are far more susceptible.",
        ),
    ),
    "mosquito": (
        "Mosquito",
        "Slender biting flies that breed in standing water. Chiefly a human health concern "
        "around farms rather than a crop pest.",
        "No direct crop damage. The concern is disease transmission to people and livestock "
        "working near stagnant water.",
        (
            "Drain or stock standing water in field channels and containers.",
            "Level low spots where irrigation water collects after each cycle.",
        ),
    ),
    "sawfly": (
        "Sawfly",
        "Wasp relatives whose caterpillar-like larvae feed on foliage; stem sawfly larvae "
        "tunnel inside stems.",
        "Defoliation from leaf-feeding species, and lodged or broken stems where stem sawfly "
        "larvae have tunnelled and weakened the plant.",
        (
            "Cut stubble low after harvest to destroy overwintering larvae in stems.",
            "Solid-stemmed varieties resist stem sawfly damage.",
        ),
    ),
    "stem_borer": (
        "Stem Borer",
        "Moth larvae that tunnel inside stems, cutting off the plant's internal transport of "
        "water and nutrients.",
        "'Dead heart' in young plants and empty white heads in mature ones. A major pest of "
        "rice, maize and sugarcane.",
        (
            "Release Trichogramma parasitoid wasps early in the crop cycle.",
            "Destroy stubble after harvest to kill larvae overwintering in stems.",
            "Avoid excess nitrogen, which makes stems more attractive to egg-laying moths.",
        ),
    ),
}

PESTS: Final[dict[str, Pest]] = {
    slug: Pest(
        slug=slug,
        name=name,
        description=description,
        damage=damage,
        pesticides=() if slug == "earthworm" else _products(slug),
        cultural_controls=controls,
        beneficial=slug == "earthworm",
    )
    for slug, (name, description, damage, controls) in _DESCRIPTIONS.items()
}


def get_pest(slug: str) -> Pest | None:
    """Look up a pest by its classifier label."""
    return PESTS.get(slug.replace(" ", "_").lower())
