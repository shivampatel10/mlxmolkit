"""Tests for embed pipeline max-iteration resolution."""

import pytest
from rdkit import Chem
from rdkit.Chem.rdDistGeom import EmbedParameters

from mlxmolkit.pipeline.driver import _resolve_max_iterations


def _test_mols():
    return [
        Chem.AddHs(Chem.MolFromSmiles("CC")),
        Chem.AddHs(Chem.MolFromSmiles("CCO")),
    ]


def _params_with_max_iterations(max_iterations: int):
    params = EmbedParameters()
    params.maxIterations = max_iterations
    return params


def test_resolve_max_iterations_explicit_override():
    params = _params_with_max_iterations(7)

    assert _resolve_max_iterations(_test_mols(), params, 3) == 3


@pytest.mark.parametrize("explicit_max_iterations", [-1, 0])
def test_resolve_max_iterations_uses_params_when_explicit_is_not_positive(
    explicit_max_iterations,
):
    params = _params_with_max_iterations(7)

    assert (
        _resolve_max_iterations(_test_mols(), params, explicit_max_iterations)
        == 7
    )


def test_resolve_max_iterations_auto_fallback():
    mols = _test_mols()
    params = _params_with_max_iterations(0)

    assert _resolve_max_iterations(mols, params, -1) == (
        10 * max(mol.GetNumAtoms() for mol in mols)
    )
