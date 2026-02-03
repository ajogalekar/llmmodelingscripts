#!/usr/bin/env python3
"""
Universal OpenMM MD (100 ps default) for protein–ligand PDBs with auto-fix & ligand params.

What it does
------------
• Reads a protein–ligand complex PDB
• Repairs common issues (nonstandard residues, missing atoms, hydrogens)
• Keeps ligands (removes only crystallographic waters)
• Detects ligands (non-protein residues, excluding waters/ions)
• Parameters: AMBER ff14SB (protein) + TIP3P (water) + OpenFF/SMIRNOFF for ligands
• Minimization -> 50 ps NPT equil -> production (default 100 ps)
• Outputs: fixed_complex.pdb, trajectory.dcd, log.csv, final.pdb

Dependencies
------------
pip install openmm pdbfixer openmmforcefields openff-toolkit rdkit-pypi

Usage
-----
python universal_md_script.py complex.pdb \
    --ph 7.0 --padding 10.0 --timestep 0.002 --temperature 300 --pressure 1.0 \
    --steps 50000 --platform auto
"""

import argparse
import io
import sys
from pathlib import Path

from pdbfixer import PDBFixer
from openmm import app, unit, Platform
import openmm as mm

# Optional deps for ligand perception/parametrization
HAS_RDKit = True
try:
    from rdkit import Chem
    from rdkit.Chem import AllChem
except Exception:
    HAS_RDKit = False

HAS_OFF = True
try:
    from openff.toolkit.topology import Molecule
    from openmmforcefields.generators import SMIRNOFFTemplateGenerator
except Exception:
    HAS_OFF = False

ION_NAMES = {
    "NA", "K", "CL", "CA", "MG", "ZN", "MN", "CU", "FE", "FE2", "CO", "NI",
    "CS", "RB", "SR", "CD", "HG", "PB", "AL", "AG"
}
WATER_RESNAMES = {"HOH", "WAT", "H2O", "TIP", "SOL"}


def log(msg: str) -> None:
    print(f"[run_md] {msg}", flush=True)


def parse_args():
    p = argparse.ArgumentParser(description="OpenMM protein–ligand MD with automatic repairs")
    p.add_argument("pdb", type=str, help="Protein–ligand complex PDB input")
    p.add_argument("--ph", type=float, default=7.0, help="pH for protonation (default 7.0)")
    p.add_argument("--padding", type=float, default=10.0, help="Solvent padding in Å (default 10.0)")
    p.add_argument("--timestep", type=float, default=0.002, help="Time step in ps (default 0.002 = 2 fs)")
    p.add_argument("--temperature", type=float, default=300.0, help="Temperature in K (default 300)")
    p.add_argument("--pressure", type=float, default=1.0, help="Pressure in atm (default 1.0)")
    p.add_argument("--steps", type=int, default=50000, help="Production steps (50000 @2fs ≈ 100 ps)")
    p.add_argument("--platform", type=str, default="auto", choices=["auto", "CUDA", "OpenCL", "CPU"],
                   help="OpenMM compute platform")
    p.add_argument("--friction", type=float, default=1.0, help="Langevin friction (ps^-1)")
    p.add_argument("--seed", type=int, default=2025, help="Random seed")
    return p.parse_args()


def write_selection_pdb(topology, positions, atom_indices):
    """Return a PDB string containing only the specified atom indices."""
    sel = set(atom_indices)
    mod = app.Modeller(topology, positions)
    mod.delete([a for a in topology.atoms() if a.index not in sel])
    s = io.StringIO()
    app.PDBFile.writeFile(mod.topology, mod.positions, s)
    return s.getvalue()


def guess_rdkit_from_pdbblock(pdb_block: str):
    """Try to build an RDKit Mol from a PDB block, guessing bonds if needed."""
    if not HAS_RDKit:
        return None
    # First try: use CONECT if present
    m = Chem.MolFromPDBBlock(pdb_block, removeHs=False, sanitize=False)
    if m is not None:
        try:
            Chem.SanitizeMol(m)
            return m
        except Exception:
            pass
    # Fallback: proximity bonding
    try:
        m2 = Chem.MolFromPDBBlock(pdb_block, removeHs=False, sanitize=False, proximityBonding=True)
        if m2 is None:
            return None
        Chem.SanitizeMol(m2)
        return m2
    except Exception:
        return None


def is_protein_residue(res):
    """Heuristic: residue is protein if it contains N-CA-C backbone atoms."""
    atom_names = {a.name.strip().upper() for a in res.atoms()}
    return {"N", "CA", "C"}.issubset(atom_names)


def collect_ligand_residues(topology):
    """Return non-water, non-ion, non-protein residues as ligands (size > 2 atoms)."""
    lig = []
    for chain in topology.chains():
        for res in chain.residues():
            name = res.name.strip().upper()
            if name in WATER_RESNAMES:
                continue
            if name in ION_NAMES:
                continue
            if is_protein_residue(res):
                continue
            if sum(1 for _ in res.atoms()) <= 2:
                continue
            lig.append(res)
    return lig


def build_forcefield_with_ligands(topology, positions, ligand_residues):
    """
    Build protein+water ForceField and register SMIRNOFF templates for ligands (if available).
    """
    ff_files = ["amber14/protein.ff14SB.xml", "amber14/tip3p.xml"]
    ff = app.ForceField(*ff_files)

    if not ligand_residues:
        return ff

    if not HAS_OFF:
        log("⚠ OpenFF toolkit not available; ligands will not get SMIRNOFF params.")
        return ff

    off_mols, failed = [], []
    log(f"Attempting OpenFF parametrization for {len(ligand_residues)} ligand residue(s).")

    for res in ligand_residues:
        idxs = [a.index for a in res.atoms()]
        pdb_block = write_selection_pdb(topology, positions, idxs)
        offmol = None

        # Strategy A: RDKit → OFF (adds Hs to improve graph fidelity)
        if HAS_RDKit:
            rdmol = guess_rdkit_from_pdbblock(pdb_block)
            if rdmol is not None:
                try:
                    rdmol = Chem.AddHs(rdmol, addCoords=True)
                    # Embed if needed (coordinates may already exist from PDB)
                    try:
                        AllChem.EmbedMolecule(rdmol, useRandomCoords=True, randomSeed=1)
                    except Exception:
                        pass
                except Exception:
                    pass
                try:
                    offmol = Molecule.from_rdkit(rdmol, allow_undefined_stereo=True)
                    offmol.name = res.name
                except Exception:
                    offmol = None

        # Strategy B: directly from OpenMM topology slice (works if bonds present)
        if offmol is None:
            try:
                offmol = Molecule.from_openmm_topology(topology, unique=False, atom_indices=idxs)
                offmol.name = res.name
            except Exception:
                offmol = None

        if offmol and offmol.n_atoms > 2:
            off_mols.append(offmol)
            log(f"  ✔ {res.name}: {offmol.n_atoms} atoms")
        else:
            failed.append(res.name)

    if off_mols:
        # Critical fix: explicitly add molecules before registering generator
        smirnoff = SMIRNOFFTemplateGenerator(forcefield="openff_unconstrained-2.0.0.offxml")
        smirnoff.add_molecules(off_mols)
        ff.registerTemplateGenerator(smirnoff.generator)
        log(f"Registered SMIRNOFF template generator for {len(off_mols)} ligand(s).")

    if failed:
        uniq_failed = sorted(set(failed))
        log(f"⚠ Could not prepare OFF molecules for: {uniq_failed}. "
            "If these are true organic ligands, supply an SDF/MOL2 or a PDB with CONECT records.")

    return ff


def main():
    args = parse_args()
    pdb_path = Path(args.pdb)
    if not pdb_path.exists():
        sys.exit(f"Input PDB not found: {pdb_path}")

    log(f"Loading and auto-fixing PDB: {pdb_path}")
    fixer = PDBFixer(filename=str(pdb_path))

    # Repair protein: normalize residues, fill gaps/atoms, add hydrogens
    log("  Finding/replacing nonstandard residues; filling gaps/atoms...")
    fixer.findMissingResidues()
    fixer.findMissingAtoms()
    fixer.findNonstandardResidues()
    fixer.replaceNonstandardResidues()
    log(f"  Adding hydrogens at pH {args.ph:.2f} ...")
    fixer.addMissingHydrogens(pH=args.ph)

    # Build modeller; remove only crystallographic waters (keep ligands)
    modeller = app.Modeller(fixer.topology, fixer.positions)
    waters = [res for res in modeller.topology.residues() if res.name.upper() in WATER_RESNAMES]
    if waters:
        modeller.delete(waters)
        log(f"  Removed {len(waters)} crystallographic water residues.")

    # Detect ligands before solvation
    lig_res = collect_ligand_residues(modeller.topology)
    log(f"Detected {len(lig_res)} ligand residue(s): {[r.name for r in lig_res] or '[]'}")

    # Build full ForceField (protein + TIP3P + SMIRNOFF ligands where possible)
    ff = build_forcefield_with_ligands(modeller.topology, modeller.positions, lig_res)

    # Solvate using the same FF (ensures residue templates exist during addSolvent)
    padding = args.padding * unit.angstroms
    log(f"  Adding TIP3P water (padding {args.padding:.1f} Å) and neutralizing with ions...")
    modeller.addSolvent(ff, model='tip3p', padding=padding, neutralize=True)

    # Save repaired/solvated starting structure
    app.PDBFile.writeFile(modeller.topology, modeller.positions, open("fixed_complex.pdb", "w"))
    log("Wrote repaired/solvated structure to fixed_complex.pdb")

    # Create system
    log("Creating System (PME, HBonds constraints)...")
    system = ff.createSystem(
        modeller.topology,
        nonbondedMethod=app.PME,
        nonbondedCutoff=1.0 * unit.nanometer,
        constraints=app.HBonds,
        rigidWater=True,
        ewaldErrorTolerance=1.0e-4,
    )
    system.addForce(mm.MonteCarloBarostat(args.pressure * unit.atmospheres,
                                          args.temperature * unit.kelvin, 50))

    # Integrator
    timestep = args.timestep * unit.picoseconds
    integrator = mm.LangevinMiddleIntegrator(
        args.temperature * unit.kelvin,
        args.friction / unit.picoseconds,
        timestep
    )
    integrator.setRandomNumberSeed(args.seed)

    # Platform selection
    if args.platform == "auto":
        platform = None
        for name in ("CUDA", "OpenCL", "CPU"):
            try:
                platform = Platform.getPlatformByName(name)
                log(f"Using platform: {name}")
                break
            except Exception:
                continue
        if platform is None:
            platform = Platform.getPlatformByName("CPU")
            log("Using platform: CPU")
    else:
        platform = Platform.getPlatformByName(args.platform)
        log(f"Using platform: {args.platform}")

    sim = app.Simulation(modeller.topology, system, integrator, platform)
    sim.context.setPositions(modeller.positions)

    # Minimization
    log("Minimizing energy...")
    sim.minimizeEnergy(maxIterations=5000)

    # Equilibration (50 ps NPT)
    log("Equilibrating (50 ps)...")
    n_eq = int(50.0 / args.timestep)  # timestep in ps
    sim.context.setVelocitiesToTemperature(args.temperature * unit.kelvin, args.seed)
    sim.reporters.append(app.StateDataReporter(
        "log.csv", 1000, step=True, potentialEnergy=True, kineticEnergy=True, totalEnergy=True,
        temperature=True, volume=True, density=True, progress=True, speed=True,
        elapsedTime=True, separator=","
    ))
    sim.step(n_eq)

    # Production
    total_ps = args.steps * args.timestep
    log(f"Production run: {args.steps} steps (~{total_ps:.1f} ps)...")
    sim.reporters.append(app.DCDReporter("trajectory.dcd", 1000))
    sim.step(args.steps)

    # Final coordinates
    state = sim.context.getState(getPositions=True, getVelocities=False)
    app.PDBFile.writeFile(modeller.topology, state.getPositions(), open("final.pdb", "w"))
    log("Done. Wrote trajectory.dcd, log.csv, and final.pdb")


if __name__ == "__main__":
    main()

