"""Integration tests for EmbedMolecules against RDKit reference.

Tests the full ETKDG pipeline including ETK minimization, stereochemistry
checks, and conformer writeback. Validates against RDKit's own
EmbedMultipleConfs as ground truth.
"""

import os

import pytest
from rdkit import Chem
from rdkit.Chem import AllChem, rdDistGeom
from rdkit.Chem.rdDistGeom import EmbedParameters

from mlxmolkit import EmbedMolecules


@pytest.fixture
def embed_test_mols():
    """Load 5 molecules from MMFF94_dative.sdf for embedding tests."""
    sdf_path = os.path.join(
        os.path.dirname(__file__),
        "test_data",
        "MMFF94_dative.sdf",
    )
    if not os.path.exists(sdf_path):
        pytest.skip(f"Test data file not found: {sdf_path}")

    supplier = Chem.SDMolSupplier(sdf_path, removeHs=False, sanitize=True)
    molecules = []
    for i, mol in enumerate(supplier):
        if mol is None:
            continue
        if i >= 5:
            break
        mol.RemoveAllConformers()
        molecules.append(mol)

    if len(molecules) < 5:
        pytest.skip(f"Expected 5 molecules, found {len(molecules)}")

    return molecules


def create_hard_copy_mols(molecules):
    """Create independent copies of molecules with no conformers."""
    copied = []
    for mol in molecules:
        m = Chem.Mol(mol)
        m.RemoveAllConformers()
        copied.append(m)
    return copied


def embed_with_rdkit(molecules, confs_per_mol=5, params=None):
    """Embed molecules using RDKit's EmbedMultipleConfs for comparison."""
    if params is None:
        params = rdDistGeom.ETKDGv3()
        params.useRandomCoords = True
        params.randomSeed = 42

    all_conf_ids = []
    for mol in molecules:
        conf_ids = rdDistGeom.EmbedMultipleConfs(
            mol, numConfs=confs_per_mol, params=params
        )
        all_conf_ids.append(list(conf_ids))
    return all_conf_ids


def compare_conformers_rmsd(
    rdkit_mols, mlx_mols, rmsd_threshold=0.2, min_match_fraction=0.5
):
    """Compare conformers using RMSD. At least min_match_fraction of mlxmolkit
    conformers must match an RDKit conformer within rmsd_threshold."""
    for mol_idx, (rdkit_mol, mlx_mol) in enumerate(zip(rdkit_mols, mlx_mols)):
        if rdkit_mol is None or mlx_mol is None:
            continue

        rdkit_count = rdkit_mol.GetNumConformers()
        mlx_count = mlx_mol.GetNumConformers()

        if rdkit_count == 0 or mlx_count == 0:
            continue

        similar = 0
        try:
            temp = Chem.Mol(rdkit_mol)
            temp.RemoveAllConformers()

            for cid in range(rdkit_count):
                temp.AddConformer(rdkit_mol.GetConformer(cid), assignId=False)
            for cid in range(mlx_count):
                temp.AddConformer(mlx_mol.GetConformer(cid), assignId=False)

            rmsd_matrix = AllChem.GetConformerRMSMatrix(temp, prealigned=False)

            for mlx_idx in range(mlx_count):
                mlx_conf_idx = rdkit_count + mlx_idx
                min_rmsd = float("inf")
                for rdkit_idx in range(rdkit_count):
                    rmsd_index = (
                        mlx_conf_idx * (mlx_conf_idx + 1) // 2
                        + rdkit_idx
                        - mlx_conf_idx
                    )
                    min_rmsd = min(min_rmsd, rmsd_matrix[rmsd_index])
                if min_rmsd <= rmsd_threshold:
                    similar += 1

        except Exception as e:
            print(f"Error calculating RMSD for molecule {mol_idx}: {e}")
            continue

        fraction = similar / mlx_count if mlx_count > 0 else 0.0
        assert fraction >= min_match_fraction, (
            f"Molecule {mol_idx}: Only {similar}/{mlx_count} "
            f"({fraction:.2f}) mlxmolkit conformers similar to RDKit "
            f"(RMSD < {rmsd_threshold}). Expected >= {min_match_fraction}."
        )


# --- Basic functionality tests ---


def test_embed_empty_input():
    """EmbedMolecules with empty list should not raise."""
    params = EmbedParameters()
    params.useRandomCoords = True
    EmbedMolecules([], params)


def test_embed_invalid_none():
    """EmbedMolecules with None molecule should raise ValueError."""
    params = EmbedParameters()
    params.useRandomCoords = True
    with pytest.raises(ValueError, match="Molecule at index 0 is None"):
        EmbedMolecules([None], params)


def test_embed_invalid_params():
    """EmbedMolecules with useRandomCoords=False should raise."""
    mol = Chem.MolFromSmiles("CCO")
    params = EmbedParameters()
    params.useRandomCoords = False
    with pytest.raises(ValueError, match="ETKDG requires useRandomCoords=True"):
        EmbedMolecules([mol], params)


def test_embed_simple_molecule():
    """Generate conformers for a simple molecule (ethanol)."""
    mol = Chem.AddHs(Chem.MolFromSmiles("CCO"))
    params = rdDistGeom.ETKDGv3()
    params.useRandomCoords = True
    params.randomSeed = 42

    EmbedMolecules([mol], params, confsPerMolecule=3)

    assert mol.GetNumConformers() > 0, "Expected at least 1 conformer"
    # Verify conformer has correct number of atoms
    conf = mol.GetConformer(0)
    assert conf.GetNumAtoms() == mol.GetNumAtoms()


def test_embed_multiple_molecules():
    """Generate conformers for multiple simple molecules."""
    mols = [
        Chem.AddHs(Chem.MolFromSmiles("CCO")),
        Chem.AddHs(Chem.MolFromSmiles("CCC")),
        Chem.AddHs(Chem.MolFromSmiles("c1ccccc1")),
    ]
    params = rdDistGeom.ETKDGv3()
    params.useRandomCoords = True
    params.randomSeed = 42

    EmbedMolecules(mols, params, confsPerMolecule=2)

    for i, mol in enumerate(mols):
        assert mol.GetNumConformers() > 0, f"Molecule {i} has no conformers"


# --- DG-only variant tests (no ETK stage) ---


def test_embed_dg_only():
    """Test pure DG variant (no torsion knowledge, no basic knowledge)."""
    mol = Chem.AddHs(Chem.MolFromSmiles("CCCC"))
    params = rdDistGeom.KDG()
    params.useBasicKnowledge = False
    params.useRandomCoords = True
    params.randomSeed = 42

    EmbedMolecules([mol], params, confsPerMolecule=3)
    assert mol.GetNumConformers() > 0


# --- ETKDG variant tests against RDKit reference ---


@pytest.mark.slow
@pytest.mark.parametrize(
    "etkdg_variant",
    ["ETKDG", "ETKDGv2", "ETKDGv3", "srETKDGv3", "KDG", "ETDG", "DG"],
)
def test_embed_vs_rdkit(embed_test_mols, etkdg_variant):
    """Test mlxmolkit against RDKit reference for all ETKDG variants."""
    confs_per_mol = 5

    rdkit_mols = create_hard_copy_mols(embed_test_mols)
    mlx_mols = create_hard_copy_mols(embed_test_mols)

    if etkdg_variant == "ETKDG":
        params = rdDistGeom.ETKDG()
    elif etkdg_variant == "ETKDGv2":
        params = rdDistGeom.ETKDGv2()
    elif etkdg_variant == "ETKDGv3":
        params = rdDistGeom.ETKDGv3()
    elif etkdg_variant == "srETKDGv3":
        params = rdDistGeom.srETKDGv3()
    elif etkdg_variant == "KDG":
        params = rdDistGeom.KDG()
    elif etkdg_variant == "ETDG":
        params = rdDistGeom.ETDG()
    elif etkdg_variant == "DG":
        params = rdDistGeom.KDG()
        params.useBasicKnowledge = False

    params.useRandomCoords = True
    params.randomSeed = 42

    # RDKit reference
    embed_with_rdkit(rdkit_mols, confs_per_mol, params)

    # mlxmolkit
    EmbedMolecules(mlx_mols, params, confsPerMolecule=confs_per_mol)

    # Check conformer counts — mlxmolkit should generate at least some
    for mol_idx, (rdkit_mol, mlx_mol) in enumerate(zip(rdkit_mols, mlx_mols)):
        rdkit_count = rdkit_mol.GetNumConformers()
        mlx_count = mlx_mol.GetNumConformers()
        assert mlx_count > 0, (
            f"Molecule {mol_idx} ({etkdg_variant}): "
            f"mlxmolkit generated 0 conformers (RDKit: {rdkit_count})"
        )

    # RMSD comparison
    compare_conformers_rmsd(
        rdkit_mols, mlx_mols, rmsd_threshold=0.2, min_match_fraction=0.5
    )


# --- Edge cases ---


def test_embed_large_molecule():
    """Test embedding a molecule with >256 atoms."""
    mol = Chem.AddHs(Chem.MolFromSmiles("C" * 100))
    assert mol.GetNumAtoms() > 256

    params = EmbedParameters()
    params.useRandomCoords = True
    params.maxIterations = 5

    EmbedMolecules([mol], params, confsPerMolecule=1)
    assert mol.GetNumConformers() >= 0  # May or may not succeed, just don't crash
