# mlxmolkit

GPU-accelerated molecular 3D coordinate generation on Apple Silicon using MLX.

mlxmolkit is a drop-in replacement for RDKit's ETKDG conformer generation that runs on Apple Silicon GPUs via the [MLX](https://github.com/ml-explore/mlx) framework. It matches the interface of NVIDIA's nvMolKit `EmbedMolecules`, bringing hardware-accelerated molecular embedding to Mac.

## Installation

```bash
pip install mlxmolkit
```

## Quick Start

```python
from rdkit import Chem
from rdkit.Chem import AllChem
from mlxmolkit import EmbedMolecules

# Prepare molecules
smiles = ["CCO", "c1ccccc1", "CC(=O)Oc1ccccc1C(=O)O"]
molecules = [Chem.AddHs(Chem.MolFromSmiles(s)) for s in smiles]

# Configure ETKDG parameters
params = AllChem.ETKDGv3()
params.useRandomCoords = True

# Generate conformers on GPU
EmbedMolecules(molecules, params, confsPerMolecule=10)
```

Supports all standard ETKDG variants: ETKDG, ETKDGv2, ETKDGv3, srETKDGv3, KDG, ETDG, and pure DG.

## Requirements

- macOS with Apple Silicon (M1/M2/M3/M4)
- Python >= 3.12

## Dependencies

- [mlx](https://github.com/ml-explore/mlx) - Apple's ML framework for Apple Silicon
- [rdkit](https://www.rdkit.org/) - Cheminformatics toolkit
- [numpy](https://numpy.org/)

## License

[MIT](LICENSE)

## Links

- [Documentation](https://shivampatel10.github.io/mlxmolkit/)
- [Repository](https://github.com/shivampatel10/mlxmolkit)
- [Issues](https://github.com/shivampatel10/mlxmolkit/issues)
