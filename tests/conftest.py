"""Shared test fixtures for mlxmolkit tests."""

import os

import pytest
from rdkit import Chem
from rdkit.Chem import AllChem, rdDistGeom


TEST_DATA_DIR = os.path.join(os.path.dirname(__file__), "test_data")
SDF_PATH = os.path.join(TEST_DATA_DIR, "MMFF94_dative.sdf")


@pytest.fixture
def ethanol_mol():
    """Ethanol (CCO) with hydrogens added."""
    mol = Chem.MolFromSmiles("CCO")
    mol = Chem.AddHs(mol)
    return mol


@pytest.fixture
def benzene_mol():
    """Benzene (c1ccccc1) with hydrogens added."""
    mol = Chem.MolFromSmiles("c1ccccc1")
    mol = Chem.AddHs(mol)
    return mol


@pytest.fixture
def propane_mol():
    """Propane (CCC) with hydrogens added."""
    mol = Chem.MolFromSmiles("CCC")
    mol = Chem.AddHs(mol)
    return mol


@pytest.fixture
def chiral_mol():
    """A molecule with a chiral center: (R)-butan-2-ol."""
    mol = Chem.MolFromSmiles("[C@@H](O)(CC)C")
    mol = Chem.AddHs(mol)
    return mol


@pytest.fixture
def sdf_mols():
    """First 5 molecules from MMFF94_dative.sdf."""
    if not os.path.exists(SDF_PATH):
        pytest.skip("MMFF94_dative.sdf not found")
    suppl = Chem.SDMolSupplier(SDF_PATH, removeHs=False)
    mols = []
    for mol in suppl:
        if mol is not None:
            mols.append(mol)
        if len(mols) >= 5:
            break
    return mols
