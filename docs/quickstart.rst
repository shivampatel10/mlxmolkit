Quick Start
===========

Basic Usage
-----------

.. code-block:: python

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

Supported ETKDG Variants
-------------------------

mlxmolkit supports all standard ETKDG variants through RDKit's
``EmbedParameters``:

- ``ETKDG``
- ``ETKDGv2``
- ``ETKDGv3``
- ``srETKDGv3``
- ``KDG``
- ``ETDG``
- Pure ``DG``

Configure the variant by passing the corresponding RDKit parameter object
to :func:`~mlxmolkit.EmbedMolecules`.
