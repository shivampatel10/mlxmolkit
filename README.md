# mlxmolkit

GPU-accelerated molecular toolkit for Apple Silicon using MLX.

mlxmolkit provides GPU-accelerated cheminformatics on Apple Silicon via the [MLX](https://github.com/ml-explore/mlx) framework. It includes conformer generation (ETKDG) and MMFF94 force field optimization, matching the interfaces of RDKit and NVIDIA's nvMolKit.

## Features

- **Conformer Generation** — Drop-in replacement for RDKit's ETKDG (`EmbedMolecules`). Supports ETKDG, ETKDGv2, ETKDGv3, srETKDGv3, KDG, ETDG, and pure DG.
- **MMFF94 Optimization** — GPU-accelerated force field optimization (`MMFFOptimizeMoleculesConfs`). All 7 MMFF energy terms with fused Metal kernel. **14x faster than RDKit** at scale.

## MMFF Optimization Performance

Benchmarked on Apple Silicon, optimizing conformers of molecules from the MMFF94 validation set:

| Conformers | Atoms | RDKit CPU | Metal GPU | Speedup |
|------------|-------|-----------|-----------|---------|
| 50 | 706 | 45 ms | 42 ms | 1.1x |
| 100 | 1,412 | 91 ms | 46 ms | 2.0x |
| 500 | 7,060 | 457 ms | 71 ms | 6.4x |
| 1,000 | 14,120 | 911 ms | 115 ms | 7.9x |
| 5,000 | 70,600 | 4,556 ms | 429 ms | 10.6x |
| 10,000 | 141,200 | 11,610 ms | 830 ms | 14.0x |

GPU throughput saturates around 12K conformers/sec. Crossover vs RDKit is ~50 conformers.

## Installation

```bash
pip install mlxmolkit
```

## Quick Start

```python
from rdkit import Chem
from rdkit.Chem import AllChem
from mlxmolkit import EmbedMolecules, MMFFOptimizeMoleculesConfs

# Prepare molecules
smiles = ["CCO", "c1ccccc1", "CC(=O)Oc1ccccc1C(=O)O"]
molecules = [Chem.AddHs(Chem.MolFromSmiles(s)) for s in smiles]

# Configure ETKDG parameters
params = AllChem.ETKDGv3()
params.useRandomCoords = True

# Generate conformers on GPU
EmbedMolecules(molecules, params, confsPerMolecule=10)

# Optimize with MMFF94 force field on GPU
energies = MMFFOptimizeMoleculesConfs(molecules)
# energies[i] = list of energies for each conformer of molecule i
```

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
