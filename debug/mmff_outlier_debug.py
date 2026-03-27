"""Debug RDKit vs MLX MMFF disagreements on fixed conformers.

Reproduces the first-20-valid-molecules benchmark and prints a detailed
breakdown for one frozen molecule/conformer.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import mlx.core as mx
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, rdDistGeom, rdForceFieldHelpers

from mlxmolkit.forcefields.mmff import (
    _angle_bend_energy,
    _bond_stretch_energy,
    _ele_energy,
    _oop_bend_energy,
    _stretch_bend_energy,
    _torsion_energy,
    _vdw_energy,
    mmff_energy_and_grad,
)
from mlxmolkit.mmff_optimize import MMFFOptimizeMoleculesConfs
from mlxmolkit.preprocessing.mmff_batching import batch_mmff_params
from mlxmolkit.preprocessing.mmff_extract import extract_mmff_params

DEFAULT_SDF = "tests/test_data/MMFF94_dative.sdf"
DEFAULT_VALID_IDX = 15
DEFAULT_CONF_IDX = 0
DEFAULT_N_MOLS = 20
DEFAULT_N_CONFS = 5
DEFAULT_SEED = 42


@dataclass
class ValidMolEntry:
    source_index: int
    mol: Chem.Mol


def _load_valid_molecules(sdf_path: str, limit: int) -> list[ValidMolEntry]:
    suppl = Chem.SDMolSupplier(sdf_path, removeHs=False)
    valid: list[ValidMolEntry] = []
    for source_index, mol in enumerate(suppl):
        if mol is None:
            continue
        if rdForceFieldHelpers.MMFFGetMoleculeProperties(mol) is None:
            continue
        valid.append(ValidMolEntry(source_index=source_index, mol=Chem.Mol(mol)))
        if len(valid) >= limit:
            break
    return valid


def _embed_copy(mol: Chem.Mol, n_confs: int, seed: int) -> Chem.Mol:
    embedded = Chem.Mol(mol)
    embedded.RemoveAllConformers()
    params = rdDistGeom.ETKDGv3()
    params.useRandomCoords = True
    params.randomSeed = seed
    params.numThreads = 1
    AllChem.EmbedMultipleConfs(embedded, numConfs=n_confs, params=params)
    return embedded


def _coords_flat(mol: Chem.Mol, conf_id: int) -> np.ndarray:
    conf = mol.GetConformer(conf_id)
    return np.array(
        [list(conf.GetAtomPosition(i)) for i in range(mol.GetNumAtoms())],
        dtype=np.float32,
    ).reshape(-1)


def _rdkit_energy(mol: Chem.Mol, conf_id: int, ignore_interfrag: bool = True) -> float:
    props = rdForceFieldHelpers.MMFFGetMoleculeProperties(mol)
    ff = rdForceFieldHelpers.MMFFGetMoleculeForceField(
        mol,
        props,
        confId=conf_id,
        ignoreInterfragInteractions=ignore_interfrag,
    )
    return ff.CalcEnergy()


def _term_breakdown(pos: mx.array, system) -> dict[str, float]:
    result = {
        "bond": 0.0,
        "angle": 0.0,
        "stretch_bend": 0.0,
        "oop": 0.0,
        "torsion": 0.0,
        "vdw": 0.0,
        "electrostatic": 0.0,
    }
    if system.bond_idx1.size > 0:
        result["bond"] = float(
            mx.sum(
                _bond_stretch_energy(
                    pos, system.bond_idx1, system.bond_idx2, system.bond_kb, system.bond_r0
                )
            )
        )
    if system.angle_idx1.size > 0:
        result["angle"] = float(
            mx.sum(
                _angle_bend_energy(
                    pos,
                    system.angle_idx1,
                    system.angle_idx2,
                    system.angle_idx3,
                    system.angle_ka,
                    system.angle_theta0,
                    system.angle_is_linear,
                )
            )
        )
    if system.sb_idx1.size > 0:
        result["stretch_bend"] = float(
            mx.sum(
                _stretch_bend_energy(
                    pos,
                    system.sb_idx1,
                    system.sb_idx2,
                    system.sb_idx3,
                    system.sb_r0_ij,
                    system.sb_r0_kj,
                    system.sb_theta0,
                    system.sb_kba_ij,
                    system.sb_kba_kj,
                )
            )
        )
    if system.oop_idx1.size > 0:
        result["oop"] = float(
            mx.sum(
                _oop_bend_energy(
                    pos,
                    system.oop_idx1,
                    system.oop_idx2,
                    system.oop_idx3,
                    system.oop_idx4,
                    system.oop_koop,
                )
            )
        )
    if system.tor_idx1.size > 0:
        result["torsion"] = float(
            mx.sum(
                _torsion_energy(
                    pos,
                    system.tor_idx1,
                    system.tor_idx2,
                    system.tor_idx3,
                    system.tor_idx4,
                    system.tor_V1,
                    system.tor_V2,
                    system.tor_V3,
                )
            )
        )
    if system.vdw_idx1.size > 0:
        result["vdw"] = float(
            mx.sum(_vdw_energy(pos, system.vdw_idx1, system.vdw_idx2, system.vdw_R_star, system.vdw_epsilon))
        )
    if system.ele_idx1.size > 0:
        result["electrostatic"] = float(
            mx.sum(
                _ele_energy(
                    pos,
                    system.ele_idx1,
                    system.ele_idx2,
                    system.ele_charge_term,
                    system.ele_diel_model,
                    system.ele_is_1_4,
                )
            )
        )
    return result


def _first_nonbonded_pairs(mol: Chem.Mol, params, limit: int) -> list[dict[str, object]]:
    frag_map = np.empty(mol.GetNumAtoms(), dtype=np.int32)
    for frag_idx, frag in enumerate(Chem.GetMolFrags(mol)):
        frag_map[list(frag)] = frag_idx

    rows: list[dict[str, object]] = []
    ele_lookup: dict[tuple[int, int], tuple[bool, float, int]] = {}
    for idx1, idx2, is_1_4, charge_term, diel_model in zip(
        params.ele_terms.idx1,
        params.ele_terms.idx2,
        params.ele_terms.is_1_4,
        params.ele_terms.charge_term,
        params.ele_terms.diel_model,
    ):
        key = (int(idx1), int(idx2))
        ele_lookup[key] = (bool(is_1_4), float(charge_term), int(diel_model))

    for idx1, idx2, r_star, epsilon in zip(
        params.vdw_terms.idx1,
        params.vdw_terms.idx2,
        params.vdw_terms.R_ij_star,
        params.vdw_terms.epsilon,
    ):
        key = (int(idx1), int(idx2))
        is_1_4, charge_term, diel_model = ele_lookup.get(key, (False, 0.0, 0))
        rows.append(
            {
                "pair": key,
                "cross_fragment": bool(frag_map[key[0]] != frag_map[key[1]]),
                "is_1_4": is_1_4,
                "R_star": float(r_star),
                "epsilon": float(epsilon),
                "charge_term": charge_term,
                "diel_model": diel_model,
            }
        )
        if len(rows) >= limit:
            break
    return rows


def reproduce_first_20(sdf_path: str, n_mols: int, n_confs: int, seed: int) -> None:
    entries = _load_valid_molecules(sdf_path, n_mols)
    print(f"Loaded {len(entries)} valid molecules from {sdf_path}")

    base = [(entry.source_index, _embed_copy(entry.mol, n_confs, seed)) for entry in entries]
    rd_mols = [Chem.Mol(mol) for _, mol in base]
    mlx_mols = [Chem.Mol(mol) for _, mol in base]

    rd_energies: list[list[float]] = []
    for mol in rd_mols:
        rdForceFieldHelpers.MMFFOptimizeMoleculeConfs(mol, maxIters=200, numThreads=1)
        rd_energies.append([
            _rdkit_energy(mol, conf_id, ignore_interfrag=True)
            for conf_id in range(mol.GetNumConformers())
        ])

    mlx_energies = MMFFOptimizeMoleculesConfs(mlx_mols, maxIters=200)

    worst: tuple[float, int, int, int, str, float, float] | None = None
    for valid_idx, ((source_index, mol), rd_vals, mlx_vals) in enumerate(
        zip(base, rd_energies, mlx_energies, strict=True)
    ):
        smiles = Chem.MolToSmiles(mol)
        for conf_id, (rd_energy, mlx_energy) in enumerate(zip(rd_vals, mlx_vals, strict=True)):
            diff = abs(mlx_energy - rd_energy)
            if worst is None or diff > worst[0]:
                worst = (
                    diff,
                    valid_idx,
                    source_index,
                    conf_id,
                    smiles,
                    mlx_energy,
                    rd_energy,
                )
    print("Worst optimized mismatch:")
    print(worst)


def debug_one(sdf_path: str, valid_idx: int, conf_id: int, n_confs: int, seed: int) -> None:
    entry = _load_valid_molecules(sdf_path, valid_idx + 1)[valid_idx]
    mol = _embed_copy(entry.mol, n_confs, seed)
    smiles = Chem.MolToSmiles(mol)

    pos_np = _coords_flat(mol, conf_id)
    pos = mx.array(pos_np)

    rd_initial = _rdkit_energy(mol, conf_id, ignore_interfrag=True)
    rd_initial_all = _rdkit_energy(mol, conf_id, ignore_interfrag=False)

    params = extract_mmff_params(mol, conf_id=conf_id)
    system = batch_mmff_params([params])
    mlx_initial, _ = mmff_energy_and_grad(pos, system)
    mx.eval(mlx_initial)
    terms = _term_breakdown(pos, system)

    rd_mol = Chem.Mol(mol)
    rdForceFieldHelpers.MMFFOptimizeMolecule(rd_mol, confId=conf_id, maxIters=200)
    rd_final = _rdkit_energy(rd_mol, conf_id, ignore_interfrag=True)

    mlx_mol = Chem.Mol(mol)
    mlx_final = MMFFOptimizeMoleculesConfs([mlx_mol], maxIters=200)[0][conf_id]

    frag_map = np.empty(mol.GetNumAtoms(), dtype=np.int32)
    for frag_idx, frag in enumerate(Chem.GetMolFrags(mol)):
        frag_map[list(frag)] = frag_idx
    vdw_cross = sum(
        1 for idx1, idx2 in zip(params.vdw_terms.idx1, params.vdw_terms.idx2, strict=True)
        if frag_map[int(idx1)] != frag_map[int(idx2)]
    )
    ele_cross = sum(
        1 for idx1, idx2 in zip(params.ele_terms.idx1, params.ele_terms.idx2, strict=True)
        if frag_map[int(idx1)] != frag_map[int(idx2)]
    )

    print(f"valid_idx={valid_idx} source_index={entry.source_index} conf_id={conf_id}")
    print(f"smiles={smiles}")
    print(f"initial_energy_rdkit_default={rd_initial:.6f}")
    print(f"initial_energy_rdkit_interfrag={rd_initial_all:.6f}")
    print(f"initial_energy_mlx={float(mlx_initial[0]):.6f}")
    print(f"final_energy_rdkit={rd_final:.6f}")
    print(f"final_energy_mlx={mlx_final:.6f}")
    print("mlx_term_breakdown:")
    for name, value in terms.items():
        print(f"  {name}: {value:.6f}")
    print("nonbonded_counts:")
    print(f"  vdw_total={len(params.vdw_terms.idx1)}")
    print(f"  vdw_cross_fragment={vdw_cross}")
    print(f"  electrostatic_total={len(params.ele_terms.idx1)}")
    print(f"  electrostatic_cross_fragment={ele_cross}")
    print(f"  electrostatic_1_4={int(np.count_nonzero(params.ele_terms.is_1_4))}")
    print("first_nonbonded_pairs:")
    for row in _first_nonbonded_pairs(mol, params, limit=20):
        print(f"  {row}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sdf", default=DEFAULT_SDF)
    parser.add_argument("--valid-idx", type=int, default=DEFAULT_VALID_IDX)
    parser.add_argument("--conf-id", type=int, default=DEFAULT_CONF_IDX)
    parser.add_argument("--n-mols", type=int, default=DEFAULT_N_MOLS)
    parser.add_argument("--n-confs", type=int, default=DEFAULT_N_CONFS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--skip-repro", action="store_true")
    args = parser.parse_args()

    if not args.skip_repro:
        reproduce_first_20(args.sdf, args.n_mols, args.n_confs, args.seed)
        print()

    debug_one(args.sdf, args.valid_idx, args.conf_id, args.n_confs, args.seed)


if __name__ == "__main__":
    main()
