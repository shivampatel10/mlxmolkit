"""Public API for ETKDG conformer generation on MLX / Apple Silicon.

Provides EmbedMolecules() as the main entry point, matching the interface
of nvMolKit's EmbedMolecules.
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from rdkit.Chem import Mol
    from rdkit.Chem.rdDistGeom import EmbedParameters

from .pipeline.driver import embed_molecules_pipeline, get_last_embed_stats

__all__ = ["EmbedMolecules", "get_last_embed_stats"]


def EmbedMolecules(
    molecules: list["Mol"],
    params: "EmbedParameters",
    confsPerMolecule: int = 1,
    maxIterations: int = -1,
    batchSize: int = 250,
) -> None:
    """Embed multiple molecules with multiple conformers using MLX on Apple Silicon.

    This function performs GPU-accelerated ETKDG conformer generation using
    Apple Silicon's Metal GPU via MLX. Input molecules are modified in-place
    with generated conformers.

    Supports all standard ETKDG variants: ETKDG, ETKDGv2, ETKDGv3,
    srETKDGv3, KDG, ETDG, and pure DG.

    Args:
        molecules: List of RDKit molecules to embed. Molecules should be
            prepared (sanitized, explicit hydrogens added if needed).
        params: RDKit EmbedParameters object with embedding settings.
            Must have useRandomCoords=True for ETKDG.
            Existing conformers are cleared unless params.clearConfs is false.
        confsPerMolecule: Number of conformers to generate per molecule.
        maxIterations: Maximum ETKDG retry iterations, -1 for automatic
            calculation (10 * max_atoms).
        batchSize: Maximum molecules per GPU batch. Larger batches are
            more efficient but may hit Metal GPU timeout limits. Default
            250 (5,000 entries at 20 confs/mol).

    Returns:
        None. Input molecules are modified in-place with generated conformers.

    Raises:
        ValueError: If any molecule is None, or if useRandomCoords is not True.

    Example:
        >>> from rdkit import Chem
        >>> from rdkit.Chem.rdDistGeom import ETKDGv3
        >>> from mlxmolkit import EmbedMolecules
        >>>
        >>> mol = Chem.AddHs(Chem.MolFromSmiles('CCO'))
        >>> params = ETKDGv3()
        >>> params.useRandomCoords = True
        >>> EmbedMolecules([mol], params, confsPerMolecule=5)
        >>> mol.GetNumConformers()  # Should be 5
    """
    # Validate input
    if not molecules:
        return

    for i, mol in enumerate(molecules):
        if mol is None:
            raise ValueError(f"Molecule at index {i} is None")

    if not params.useRandomCoords:
        raise ValueError("ETKDG requires useRandomCoords=True in EmbedParameters")

    if getattr(params, "clearConfs", True):
        for mol in molecules:
            mol.RemoveAllConformers()

    # Process in chunks to avoid Metal GPU timeout on large batches
    for chunk_start in range(0, len(molecules), batchSize):
        chunk = molecules[chunk_start : chunk_start + batchSize]
        embed_molecules_pipeline(
            chunk,
            params,
            confs_per_mol=confsPerMolecule,
            max_iterations=maxIterations,
        )
