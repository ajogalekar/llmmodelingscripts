This repository contains a collection of practical molecular modeling scripts that I use in day-to-day structure-based workflows.
Many of these were written or refined with assistance from large language models (ChatGPT and Claude), then iterated and tested by hand.

They are shared as-is, primarily for transparency, reuse, and adaptation by others.

What’s in here:

The scripts focus on ligand preparation, docking, and lightweight MD, with an emphasis on being:

pragmatic rather than elegant

robust to messy real-world inputs

usable as standalone command-line tools

Included scripts:

pdbfix_universal_v2.py
Robust PDB preprocessing using PDBFixer and OpenMM.
Handles missing atoms/residues, protonation, optional water/heterogen removal, and residue renumbering.

rdkit_conformer_generator.py
Simple, reliable RDKit-based conformer generation with MMFF optimization and optional RMSD clustering.

split_sdf.py
Splits large multi-compound SDF files into individual ligand SDFs with clean filenames.
Useful for ChEMBL or vendor downloads.

universal_docking_script_final.py
General protein–ligand docking workflow using RDKit + AutoDock Vina.
Automatically detects binding sites from co-crystallized ligands when present.

combined_conf_docking.py
Generates multiple ligand conformers and rigidly docks each one, reporting the best-scoring pose per ligand.

self_docking_script.py
Convenience script for self-docking / redocking ligands from an existing protein–ligand complex.

universal_md_script.py
Short OpenMM molecular dynamics workflow for protein–ligand complexes with automatic fixing and ligand parametrization (AMBER + OpenFF where available).

Intended audience:

These scripts are likely most useful for:

computational chemists and modelers

medicinal chemistry teams doing quick docking triage

students or researchers who want working examples, not frameworks

They are not intended to be:

production-grade software

rigorously benchmarked methods

drop-in replacements for commercial platforms
