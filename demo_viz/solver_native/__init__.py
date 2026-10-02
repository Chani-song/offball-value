"""The current solver's own release quantities, from the research code.

`_passer2on1_payoff.py` is byte-identical to origin/kyuhyeok-dev's
andrew-passer2on1/passer2on1/payoff.py so it can still be diffed against it.
"""

from .release import (
    GROUND,
    PANEL_FEATURES,
    Availability,
    availability,
    release_quantities,
    solver_release_targets,
)

__all__ = ["GROUND", "PANEL_FEATURES", "Availability", "availability",
           "release_quantities", "solver_release_targets"]
