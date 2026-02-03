#!/usr/bin/env python3
"""
pdbfix_universal_v2_multi.py — robust PDB preprocessor using PDBFixer + OpenMM
Modified to accept multiple PDB IDs or files as input.

Usage:
  python pdbfix_universal_v2_multi.py INPUT1 [INPUT2 ...] [options]

Examples:
  python pdbfix_universal_v2_multi.py 4RZV 1crn.pdb --drop-water
  python pdbfix_universal_v2_multi.py 4MQT 5T04 6PS0 7F1Z --ph 7.4 --renumber global
"""

import argparse
import sys
import tempfile
from pathlib import Path

from pdbfixer import PDBFixer
from openmm.app import PDBFile, Topology
from openmm import unit

# --- helpers from your original script unchanged ---
def _is_pdb_id(s: str) -> bool:
    return len(s) == 4 and s.isalnum()

def _normalize_altlocs_text(in_path: Path, out_path: Path, policy: str = "first"):
    keep_codes = [" ", "A"]
    with open(in_path, "r") as fin, open(out_path, "w") as fout:
        for line in fin:
            if line.startswith(("ATOM  ", "HETATM")) and len(line) >= 27:
                altloc = line[16]
                if policy == "first" and altloc not in keep_codes:
                    continue
                if policy == "first" and altloc != " ":
                    line = line[:16] + " " + line[17:]
            fout.write(line)

def _run_pdbfixer(input_arg: str,
                  ph: float,
                  drop_water: bool,
                  drop_hets: bool,
                  add_terminal_missing_residues: bool):
    if Path(input_arg).exists():
        fixer = PDBFixer(filename=str(input_arg))
    elif _is_pdb_id(input_arg):
        fixer = PDBFixer(pdbid=input_arg)
    else:
        raise ValueError(f"INPUT not found or not a 4-char PDB ID: {input_arg}")

    fixer.findNonstandardResidues()
    fixer.replaceNonstandardResidues()

    fixer.findMissingResidues()
    if not add_terminal_missing_residues:
        fixer.missingResidues = {}
    else:
        term_only = {}
        for (chain, idx), resname in fixer.missingResidues.items():
            if idx == 0 or idx == len(chain.residues()) - 1:
                term_only[(chain, idx)] = resname
        fixer.missingResidues = term_only

    fixer.findMissingAtoms()
    fixer.addMissingAtoms()

    if drop_hets:
        fixer.removeHeterogens(True)
    else:
        if drop_water:
            fixer.removeHeterogens(False)

    fixer.addMissingHydrogens(pH=ph)

    return fixer

def _renumber_residues(fixer: PDBFixer, mode: str = "per-chain"):
    old_top = fixer.topology
    positions = fixer.positions

    new_top = Topology()
    atom_map = {}

    if mode not in ("per-chain", "global"):
        mode = "per-chain"

    global_counter = 1
    for chain in old_top.chains():
        new_chain = new_top.addChain(chain.id)
        res_counter = 1 if mode == "per-chain" else global_counter

        for residue in chain.residues():
            new_res = new_top.addResidue(residue.name, new_chain, id=str(res_counter))
            res_counter += 1
            if mode == "global":
                global_counter += 1

            for atom in residue.atoms():
                new_atom = new_top.addAtom(atom.name, atom.element, new_res)
                atom_map[atom] = new_atom

    for a0, a1 in old_top.bonds():
        if a0 in atom_map and a1 in atom_map:
            new_top.addBond(atom_map[a0], atom_map[a1])

    pos_unit = positions.unit if hasattr(positions, 'unit') else unit.angstrom
    new_positions_values = [positions[atom.index]._value if hasattr(positions[atom.index], '_value')
                            else positions[atom.index] for atom in old_top.atoms()]
    new_positions = new_positions_values * pos_unit

    return new_top, new_positions

def process_input(in_arg, args):
    tmp_in = None
    if Path(in_arg).exists() and args.altloc == "first":
        tmp_in = Path(tempfile.mkstemp(suffix=".pdb", prefix="altloc_")[1])
        _normalize_altlocs_text(Path(in_arg), tmp_in, policy="first")
        in_arg = str(tmp_in)

    fixer = _run_pdbfixer(
        input_arg=in_arg,
        ph=args.ph,
        drop_water=args.drop_water,
        drop_hets=args.drop_hets,
        add_terminal_missing_residues=args.add_terminal_missing_residues,
    )

    if args.renumber != "no":
        new_top, new_pos = _renumber_residues(fixer, mode=args.renumber)
        fixer.topology = new_top
        fixer.positions = new_pos

    if args.output:
        outdir = Path(args.output)
        outdir.mkdir(parents=True, exist_ok=True)
        out_path = outdir / f"{Path(in_arg).stem}_fixed.pdb"
    else:
        base = Path(in_arg).stem if Path(in_arg).exists() else in_arg
        out_path = Path(f"{base}_fixed.pdb")

    with open(out_path, "w") as f:
        PDBFile.writeFile(fixer.topology, fixer.positions, f)

    num_atoms = sum(1 for _ in fixer.topology.atoms())
    num_residues = sum(1 for _ in fixer.topology.residues())
    num_chains = sum(1 for _ in fixer.topology.chains())
    print(f"Saved: {out_path}")
    print(f"Chains: {num_chains} | Residues: {num_residues} | Atoms: {num_atoms}")
    print(f"Options: pH={args.ph}, drop_water={args.drop_water}, drop_hets={args.drop_hets}, renumber={args.renumber}, altloc={args.altloc}")

    if tmp_in and Path(tmp_in).exists():
        try:
            Path(tmp_in).unlink()
        except Exception:
            pass

def main():
    ap = argparse.ArgumentParser(description="Universal PDB fixer/normalizer (PDBFixer + OpenMM)")
    ap.add_argument("inputs", nargs="+", help="PDB/mmCIF paths or 4-char PDB IDs (multiple allowed)")
    ap.add_argument("-o", "--output", help="Output directory (default: current dir)", default=None)
    ap.add_argument("--ph", type=float, default=7.0, help="pH for adding hydrogens (default: 7.0)")
    ap.add_argument("--altloc", choices=["keep", "first"], default="keep",
                    help="Handle altlocs: 'first' keeps blank/'A' only")
    ap.add_argument("--drop-water", action="store_true", help="Remove crystallographic waters")
    ap.add_argument("--drop-hets", action="store_true", help="Remove ALL heterogens (incl. ligands)")
    ap.add_argument("--add-terminal-missing-residues", action="store_true",
                    help="Add missing residues at chain termini")
    ap.add_argument("--renumber", choices=["per-chain", "global", "no"], default="per-chain",
                    help="Renumber residues (default: per-chain)")
    args = ap.parse_args()

    for in_arg in args.inputs:
        process_input(in_arg, args)

if __name__ == "__main__":
    main()

