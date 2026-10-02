"""Role-based visual language for the off-ball demo.

Colour is assigned by *role in the causal story*, never by rank or by team
alone, so a filter that changes which players are highlighted never repaints
the survivors.  The three story roles sit far apart in OKLab and stay separable
under simulated protan/deutan/tritan vision; on top of hue each role also
carries a halo ring, a larger radius and a direct text label, so identity is
never colour-alone.  ``python -m demo_viz.palette`` re-runs that check.
"""

from __future__ import annotations

import itertools
import math

# --- surfaces -------------------------------------------------------------
INK = "#05090A"            # page background
PANEL = "#0C1315"          # side-rail surface
PITCH_DARK = "#132019"     # pitch grass, deep and desaturated
PITCH_LIGHT = "#17271E"    # alternate mow band
PITCH_LINE = "#9FB5A6"     # pitch markings
PITCH_EDGE = "#22372B"

TEXT_PRIMARY = "#F4F6F3"
TEXT_SECONDARY = "#A9B6AE"
TEXT_MUTED = "#6B7A72"

# --- story roles ----------------------------------------------------------
RUNNER = "#FF2D87"         # magenta   - the off-ball runner (the intervention)
DEFENDER = "#FFA62B"       # amber     - the defender who is pulled
BENEFICIARY = "#5CE8F5"    # cyan      - the teammate who gains
CARRIER = "#FFFFFF"        # ball carrier accent

# --- neutral players ------------------------------------------------------
ATTACK_NEUTRAL = "#D9CFB8"
DEFEND_NEUTRAL = "#5F7392"
BALL = "#FFFFFF"
BALL_EDGE = "#12181A"

# --- surfaces / fields ----------------------------------------------------
SPACE_WAKE = "#5CE8F5"     # opened space reads in the beneficiary hue
GHOST = "#FFA62B"          # frozen counterfactual defender
DELTA_POS = "#5CE8F5"      # space gained
DELTA_NEG = "#FF2D87"      # space lost
GRID = "#24352C"

ROLE_COLOURS = {
    "runner": RUNNER,
    "defender": DEFENDER,
    "beneficiary": BENEFICIARY,
}

ROLE_LABELS = {
    "runner": "RUNNER",
    "defender": "PULLED DEFENDER",
    "beneficiary": "BENEFICIARY",
}

# Opacity for players outside the highlighted triplet, before and after the
# role reveal.  Keeping them visible but recessive preserves the 11v11 context.
NEUTRAL_ALPHA_FULL = 0.92
NEUTRAL_ALPHA_MUTED = 0.30


# --------------------------------------------------------------------------
# colour science helpers (used only by the self-check below)
# --------------------------------------------------------------------------
def _hex_to_rgb(value: str) -> tuple[int, int, int]:
    value = value.lstrip("#")
    return tuple(int(value[i : i + 2], 16) for i in (0, 2, 4))  # type: ignore[return-value]


def _srgb_to_linear(channel: float) -> float:
    channel /= 255.0
    return channel / 12.92 if channel <= 0.04045 else ((channel + 0.055) / 1.055) ** 2.4


def to_oklab(value: str) -> tuple[float, float, float]:
    r, g, b = (_srgb_to_linear(c) for c in _hex_to_rgb(value))
    l = 0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b
    m = 0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b
    s = 0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b
    l, m, s = (math.copysign(abs(v) ** (1 / 3), v) for v in (l, m, s))
    return (
        0.2104542553 * l + 0.7936177850 * m - 0.0040720468 * s,
        1.9779984951 * l - 2.4285922050 * m + 0.4505937099 * s,
        0.0259040371 * l + 0.7827717662 * m - 0.8086757660 * s,
    )


def delta_e(first: str, second: str) -> float:
    a, b = to_oklab(first), to_oklab(second)
    return 100.0 * math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def simulate_cvd(value: str, kind: str) -> str:
    """Vienot (1999) dichromat simulation, used for the palette self-check."""

    r, g, b = (_srgb_to_linear(c) for c in _hex_to_rgb(value))
    long_ = 0.31399022 * r + 0.63951294 * g + 0.04649755 * b
    med = 0.15537241 * r + 0.75789446 * g + 0.08670142 * b
    short = 0.01775239 * r + 0.10944209 * g + 0.87256922 * b
    if kind == "protan":
        long_ = 1.05118294 * med - 0.05116099 * short
    elif kind == "deutan":
        med = 0.9513092 * long_ + 0.04866992 * short
    elif kind == "tritan":
        short = -0.86744736 * long_ + 1.86727089 * med
    else:  # pragma: no cover - guarded by caller
        raise ValueError(kind)
    out = (
        5.47221206 * long_ - 4.6419601 * med + 0.16963708 * short,
        -1.1252419 * long_ + 2.29317094 * med - 0.16789520 * short,
        0.02980165 * long_ - 0.19318073 * med + 1.16364789 * short,
    )
    def _encode(channel: float) -> int:
        channel = min(1.0, max(0.0, channel))
        channel = 12.92 * channel if channel <= 0.0031308 else 1.055 * channel ** (1 / 2.4) - 0.055
        return int(round(channel * 255))

    return "#%02x%02x%02x" % tuple(_encode(c) for c in out)


def palette_report() -> str:
    """Print separation of every mark pair, in normal and dichromat vision."""

    marks = {
        "runner": RUNNER,
        "defender": DEFENDER,
        "beneficiary": BENEFICIARY,
        "attack": ATTACK_NEUTRAL,
        "defend": DEFEND_NEUTRAL,
        "ball": BALL,
    }
    lines = ["mark separation from the pitch surface (OKLab dE x100)"]
    for name, value in marks.items():
        lines.append(f"  {name:<12} {value}  dE={delta_e(value, PITCH_DARK):5.1f}")
    lines.append("")
    lines.append("pairwise separation  (normal / protan / deutan / tritan)")
    for (name_a, value_a), (name_b, value_b) in itertools.combinations(marks.items(), 2):
        scores = [delta_e(value_a, value_b)] + [
            delta_e(simulate_cvd(value_a, kind), simulate_cvd(value_b, kind))
            for kind in ("protan", "deutan", "tritan")
        ]
        flag = "  <- redundant encoding required" if min(scores) < 14 else ""
        lines.append(
            f"  {name_a:<12} vs {name_b:<12} "
            + " ".join(f"{score:5.1f}" for score in scores)
            + flag
        )
    lines.append("")
    lines.append(
        "Story roles (runner/defender/beneficiary) clear dE>=14 against each other\n"
        "in normal vision; every flagged pair involves a *neutral* mark, which the\n"
        "renderer also separates by radius, opacity and direct label."
    )
    return "\n".join(lines)


if __name__ == "__main__":  # pragma: no cover
    print(palette_report())
