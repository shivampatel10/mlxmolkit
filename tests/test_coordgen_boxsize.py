"""Focused tests for coordgen box size semantics."""

import numpy as np

from mlxmolkit.pipeline.context import create_pipeline_context
from mlxmolkit.pipeline.stage_coordgen import stage_coordgen


def test_negative_box_size_mult_uses_absolute_box_size(ethanol_mol):
    ctx = create_pipeline_context([ethanol_mol])

    stage_coordgen(ctx, seed=42, box_size_mult=-10.0)

    pos = np.array(ctx.positions)
    assert np.all(pos >= -5.0)
    assert np.all(pos <= 5.0)


def test_positive_box_size_mult_scales_default_box_size(ethanol_mol):
    ctx = create_pipeline_context([ethanol_mol])

    stage_coordgen(ctx, seed=42, box_size_mult=1.0)

    pos = np.array(ctx.positions)
    assert np.all(pos >= -2.5)
    assert np.all(pos <= 2.5)
