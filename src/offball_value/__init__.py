"""Draft utilities for attacking off-ball value experiments."""

from .baseline import compute_toy_option_score, counterfactual_runner_delta
from .possession import completed_passes_from_events, infer_ball_carrier, is_controlled_by
