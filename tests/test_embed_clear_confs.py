"""Regression tests for EmbedMolecules clearConfs behavior."""

from rdkit import Chem
from rdkit.Chem import rdDistGeom

from mlxmolkit import EmbedMolecules


def _make_ethanol():
    return Chem.AddHs(Chem.MolFromSmiles("CCO"))


def _make_params(clear_confs=None):
    params = rdDistGeom.ETKDGv3()
    params.useRandomCoords = True
    params.randomSeed = 42
    if clear_confs is not None:
        params.clearConfs = clear_confs
    return params


def test_embed_repeated_default_call_clears_existing_conformers():
    mol = _make_ethanol()
    params = _make_params()

    EmbedMolecules([mol], params, confsPerMolecule=2)
    assert mol.GetNumConformers() == 2

    EmbedMolecules([mol], params, confsPerMolecule=2)
    assert mol.GetNumConformers() == 2


def test_embed_clear_confs_false_appends_existing_conformers():
    mol = _make_ethanol()
    params = _make_params(clear_confs=False)

    EmbedMolecules([mol], params, confsPerMolecule=2)
    assert mol.GetNumConformers() == 2

    EmbedMolecules([mol], params, confsPerMolecule=2)
    assert mol.GetNumConformers() == 4
