#!/usr/bin/env python3
"""
COMBINED CONFORMER GENERATION AND RIGID DOCKING SCRIPT
====================================================

This script combines conformer generation with rigid docking:
1. Generates multiple conformers for each ligand using RDKit
2. Rigidly docks each conformer using AutoDock Vina
3. Reports best scoring conformer for each ligand

Usage:
    python combined_docking.py protein.pdb ligands.sdf
    python combined_docking.py protein.pdb ligands_directory/

Requirements:
    conda install -c conda-forge rdkit autodock-vina openbabel
    pip install biopython numpy meeko
"""

import os
import sys
import subprocess
from pathlib import Path
import numpy as np
from Bio.PDB import PDBParser, PDBIO, Select
import time
import signal
import argparse
from tempfile import NamedTemporaryFile

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem, Descriptors, SaltRemover, rdMolDescriptors, rdMolAlign
    RDKIT_AVAILABLE = True
except ImportError:
    print("ERROR: RDKit not available. Install with:")
    print("conda install -c conda-forge rdkit")
    RDKIT_AVAILABLE = False

try:
    from meeko import MoleculePreparation, PDBQTWriterLegacy
    MEEKO_AVAILABLE = True
except ImportError:
    print("WARNING: Meeko not available. Will use fallback method.")
    print("Install with: pip install meeko")
    MEEKO_AVAILABLE = False

class TimeoutException(Exception):
    pass

def timeout_handler(signum, frame):
    raise TimeoutException("Operation timed out")

class ProteinOnlySelect(Select):
    def accept_residue(self, residue):
        # Exclude water molecules
        if residue.get_resname() in ['HOH', 'WAT', 'H2O', 'DOD', 'D2O']:
            return False
        
        # Only standard amino acids
        standard_aa = ['ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY', 
                      'HIS', 'ILE', 'LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER', 
                      'THR', 'TRP', 'TYR', 'VAL']
        return residue.get_resname() in standard_aa

def check_dependencies():
    """Check for required software"""
    if not RDKIT_AVAILABLE:
        return False
    
    missing = []
    
    try:
        result = subprocess.run(['vina', '--version'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if result.returncode == 0:
            print("[OK] AutoDock Vina: Found")
        else:
            missing.append("vina")
    except FileNotFoundError:
        missing.append("vina")
    
    if missing:
        print(f"ERROR: Missing {', '.join(missing)}")
        print("Install with: conda install -c conda-forge autodock-vina")
        return False
    
    print("[OK] RDKit: Available")
    if MEEKO_AVAILABLE:
        print("[OK] Meeko: Available (recommended)")
    else:
        print("[WARNING] Meeko: Not available (using fallback)")
    
    return True

def find_binding_site_from_ligand(protein_file):
    """
    Detect binding site from existing ligand in structure.
    This finds HETATM records that are not water or standard amino acids.
    """
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("protein", protein_file)
    
    ligand_coords = []
    ligand_name = None
    
    # Standard amino acids to exclude
    standard_aa = ['ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY', 
                   'HIS', 'ILE', 'LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER', 
                   'THR', 'TRP', 'TYR', 'VAL']
    
    # Waters to exclude
    waters = ['HOH', 'WAT', 'H2O', 'DOD', 'D2O']
    
    # Common caps/modifications to exclude
    caps = ['ACE', 'NMA', 'NME', 'NH2']
    
    for model in structure:
        for chain in model:
            for residue in chain:
                resname = residue.get_resname()
                
                # Check if it's a ligand (not protein, not water, not cap)
                if resname not in standard_aa and resname not in waters and resname not in caps:
                    if ligand_name is None:
                        ligand_name = resname
                    
                    for atom in residue:
                        ligand_coords.append(atom.coord)
    
    if ligand_coords:
        coords = np.array(ligand_coords)
        center = np.mean(coords, axis=0)
        
        # Calculate appropriate box size based on ligand dimensions
        ligand_range = coords.max(axis=0) - coords.min(axis=0)
        max_dimension = ligand_range.max()
        
        # Box should be at least 15Å larger than ligand in each dimension
        # but not smaller than 20Å or larger than 30Å
        box_size = max(20.0, min(30.0, max_dimension + 15.0))
        
        print(f"  Found ligand '{ligand_name}' with {len(ligand_coords)} atoms")
        print(f"  Ligand dimensions: {ligand_range[0]:.1f} x {ligand_range[1]:.1f} x {ligand_range[2]:.1f} Å")
        print(f"  Box center: ({center[0]:.2f}, {center[1]:.2f}, {center[2]:.2f})")
        print(f"  Box size: {box_size:.1f} Å")
        
        return center, box_size
    else:
        print("  WARNING: No ligand found in structure!")
        print("  Falling back to protein geometric center")
        return find_binding_site_fallback(protein_file)

def find_binding_site_fallback(protein_file):
    """Fallback: use protein geometric center"""
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("protein", protein_file)
    
    coords = []
    
    for model in structure:
        for chain in model:
            for residue in chain:
                if residue.get_resname() in ['ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY', 
                                           'HIS', 'ILE', 'LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER', 
                                           'THR', 'TRP', 'TYR', 'VAL']:
                    for atom in residue:
                        coords.append(atom.coord)
    
    coords = np.array(coords)
    center = np.mean(coords, axis=0)
    box_size = 25.0
    
    return center, box_size

def extract_protein_from_complex(input_file, output_file):
    """Extract clean protein"""
    try:
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure("complex", input_file)
        
        io = PDBIO()
        io.set_structure(structure)
        io.save(str(output_file), ProteinOnlySelect())
        
        return output_file.exists() and output_file.stat().st_size > 1000
    except Exception as e:
        print(f"Protein extraction failed: {e}")
        return False

def create_protein_pdbqt(pdb_file, pdbqt_file):
    """Create protein PDBQT using obabel"""
    cmd = ["obabel", str(pdb_file), "-O", str(pdbqt_file), 
           "-xr", "-h", "-p", "7.4"]
    
    try:
        print("  Converting protein to PDBQT...", end=" ")
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=120)
        
        if result.returncode == 0 and pdbqt_file.exists():
            with open(pdbqt_file, 'r') as f:
                content = f.read()
                atom_lines = [line for line in content.split('\n') if line.startswith('ATOM')]
                
                if len(atom_lines) > 50:
                    print(f"OK ({len(atom_lines)} atoms)")
                    return True
                else:
                    print(f"FAILED (only {len(atom_lines)} atoms)")
                    return False
        else:
            print("FAILED")
            return False
            
    except subprocess.TimeoutExpired:
        print("FAILED (timeout)")
        return False

class ConformerGenerator:
    """Conformer generation functionality"""
    
    def __init__(self):
        self.molecule = None
        self.conformer_energies = []
    
    def load_molecule(self, sdf_file, mol_index=0):
        """Load molecule from SDF file"""
        try:
            suppl = Chem.SDMolSupplier(str(sdf_file), removeHs=False, sanitize=False)
            
            mol = None
            for idx, m in enumerate(suppl):
                if idx == mol_index:
                    mol = m
                    break
            
            if mol is None:
                return None
            
            # Sanitize molecule
            try:
                Chem.SanitizeMol(mol)
            except:
                return None
            
            self.molecule = mol
            return mol
            
        except Exception as e:
            return None
    
    def load_molecule_from_mol_object(self, mol):
        """Load molecule from RDKit Mol object"""
        try:
            # Sanitize molecule
            try:
                Chem.SanitizeMol(mol)
            except:
                return None
            
            self.molecule = mol
            return mol
            
        except Exception as e:
            return None
    
    def generate_conformers(self, n_conformers=50, method='ETKDG', timeout_per_conf=30):
        """Generate conformers with timeout protection"""
        if self.molecule is None:
            return 0
        
        # Clear existing conformers
        self.molecule.RemoveAllConformers()
        
        # Set up timeout for entire conformer generation
        try:
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(timeout_per_conf * 5)  # 5x timeout for safety
            
            # Generate conformers using method-specific approach
            if method == 'ETKDG':
                conf_ids = AllChem.EmbedMultipleConfs(
                    self.molecule,
                    numConfs=n_conformers,
                    randomSeed=42,
                    useExpTorsionAnglePrefs=True,
                    useBasicKnowledge=True,
                    enforceChirality=True,
                    numThreads=1
                )
            elif method == 'ETKDGv3':
                try:
                    params = AllChem.ETKDGv3()
                    params.randomSeed = 42
                    params.numThreads = 1
                    conf_ids = AllChem.EmbedMultipleConfs(
                        self.molecule,
                        numConfs=n_conformers,
                        params=params
                    )
                except AttributeError:
                    # Fallback to ETKDG
                    conf_ids = AllChem.EmbedMultipleConfs(
                        self.molecule,
                        numConfs=n_conformers,
                        randomSeed=42,
                        useExpTorsionAnglePrefs=True,
                        useBasicKnowledge=True,
                        enforceChirality=True,
                        numThreads=1
                    )
            else:
                # Basic distance geometry
                conf_ids = AllChem.EmbedMultipleConfs(
                    self.molecule,
                    numConfs=n_conformers,
                    randomSeed=42,
                    numThreads=1
                )
            
            signal.alarm(0)  # Cancel timeout
            
        except TimeoutException:
            signal.alarm(0)
            return 0
        except Exception:
            signal.alarm(0)
            return 0
        
        if len(conf_ids) == 0:
            return 0
        
        # Optimize conformers with timeout
        try:
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(timeout_per_conf * 2)  # 2x timeout for optimization
            
            # Try bulk optimization
            try:
                results = AllChem.MMFFOptimizeMolConfs(
                    self.molecule,
                    maxIters=200,
                    numThreads=1
                )
            except AttributeError:
                # Individual optimization fallback
                results = []
                for conf_id in conf_ids:
                    try:
                        mp = AllChem.MMFFGetMoleculeProperties(self.molecule, mmffVariant='MMFF94')
                        if mp is None:
                            results.append((1, float('inf')))
                            continue
                        
                        ff = AllChem.MMFFGetMoleculeForceField(self.molecule, mp, confId=conf_id)
                        if ff is None:
                            results.append((1, float('inf')))
                            continue
                        
                        converged = ff.Minimize(maxIts=200)
                        energy = ff.CalcEnergy()
                        results.append((converged, energy))
                        
                    except Exception:
                        results.append((1, float('inf')))
            
            signal.alarm(0)  # Cancel timeout
            
        except TimeoutException:
            signal.alarm(0)
            # Use unoptimized conformers
            results = [(0, 0.0) for _ in conf_ids]
        except Exception:
            signal.alarm(0)
            results = [(0, 0.0) for _ in conf_ids]
        
        # Filter successful conformers
        successful_conformers = []
        self.conformer_energies = []
        
        for i, (converged, energy) in enumerate(results):
            if energy != float('inf'):
                successful_conformers.append(conf_ids[i])
                self.conformer_energies.append(energy)
        
        if len(successful_conformers) == 0:
            return 0
        
        # Remove failed conformers
        conf_ids_to_remove = set(conf_ids) - set(successful_conformers)
        for conf_id in sorted(conf_ids_to_remove, reverse=True):
            self.molecule.RemoveConformer(conf_id)
        
        return len(successful_conformers)
    
    def get_conformers(self):
        """Get list of conformer molecules"""
        if self.molecule is None:
            return []
        
        conformers = []
        for conf_id in range(self.molecule.GetNumConformers()):
            mol_copy = Chem.Mol(self.molecule)
            mol_copy.RemoveAllConformers()
            mol_copy.AddConformer(self.molecule.GetConformer(conf_id))
            conformers.append((mol_copy, conf_id, self.conformer_energies[conf_id] if conf_id < len(self.conformer_energies) else 0.0))
        
        return conformers

def convert_conformer_to_pdbqt(mol, pdbqt_file, use_meeko=True):
    """Convert a single conformer to PDBQT format"""
    try:
        # Try Meeko first
        if use_meeko and MEEKO_AVAILABLE:
            try:
                signal.signal(signal.SIGALRM, timeout_handler)
                signal.alarm(30)  # 30 second timeout for Meeko
                
                preparator = MoleculePreparation()
                preparator.hydrate = False
                mol_setups = preparator.prepare(mol)
                
                if mol_setups and len(mol_setups) > 0:
                    setup = mol_setups[0]
                    writer = PDBQTWriterLegacy()
                    pdbqt_string = writer.write_string(setup)
                    
                    if isinstance(pdbqt_string, tuple):
                        pdbqt_string = pdbqt_string[0]
                    
                    with open(pdbqt_file, 'w') as f:
                        f.write(pdbqt_string)
                    
                    signal.alarm(0)
                    return True
                
                signal.alarm(0)
                
            except TimeoutException:
                signal.alarm(0)
            except Exception:
                signal.alarm(0)
        
        # Fallback: Use temporary PDB then obabel conversion
        try:
            temp_pdb = pdbqt_file.with_suffix('.temp.pdb')
            Chem.MolToPDBFile(mol, str(temp_pdb))
            
            cmd = ["obabel", str(temp_pdb), "-O", str(pdbqt_file), "-xh"]
            result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=30)
            
            if temp_pdb.exists():
                temp_pdb.unlink()
            
            if result.returncode == 0 and pdbqt_file.exists():
                return True
            else:
                return False
                
        except subprocess.TimeoutExpired:
            if temp_pdb.exists():
                temp_pdb.unlink()
            return False
        except Exception:
            return False
            
    except Exception:
        return False

def run_docking(receptor_pdbqt, ligand_pdbqt, center, box_size, output_pdbqt):
    """Run docking with single conformer"""
    cmd = [
        "vina",
        "--receptor", str(receptor_pdbqt),
        "--ligand", str(ligand_pdbqt),
        "--out", str(output_pdbqt),
        "--center_x", f"{center[0]:.3f}",
        "--center_y", f"{center[1]:.3f}",
        "--center_z", f"{center[2]:.3f}",
        "--size_x", str(box_size),
        "--size_y", str(box_size),
        "--size_z", str(box_size),
        "--num_modes", "1",  # Only need best pose for each conformer
        "--exhaustiveness", "8",  # Reduced since we're doing many conformers
        "--energy_range", "3"
    ]
    
    try:
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=180)  # 3 min timeout
        
        if result.returncode == 0 and output_pdbqt.exists():
            # Extract best score
            for line in result.stdout.split('\n'):
                if line.strip() and line.strip()[0].isdigit():
                    parts = line.split()
                    if len(parts) >= 2:
                        try:
                            score = float(parts[1])
                            return score, True
                        except (ValueError, IndexError):
                            pass
            return None, False
        else:
            return None, False
            
    except subprocess.TimeoutExpired:
        if output_pdbqt.exists():
            output_pdbqt.unlink()
        return None, False

def find_ligand_files(ligands_input):
    """Find SDF ligand files - handles both single file and directory"""
    ligands_path = Path(ligands_input)
    
    if not ligands_path.exists():
        return []
    
    if ligands_path.is_file() and ligands_path.suffix == '.sdf':
        if ligands_path.stat().st_size > 200:
            return [ligands_path]
        else:
            return []
    
    if ligands_path.is_dir():
        sdf_files = list(ligands_path.glob("*.sdf"))
        valid_files = []
        for f in sdf_files:
            if f.stat().st_size > 200:
                valid_files.append(f)
        return sorted(valid_files)
    
    return []

def main():
    parser = argparse.ArgumentParser(description="Combined conformer generation and rigid docking")
    parser.add_argument("protein", help="Protein PDB file")
    parser.add_argument("ligands", help="Ligands SDF file or directory")
    parser.add_argument("-n", "--conformers", type=int, default=50, 
                       help="Number of conformers per ligand (default: 50)")
    parser.add_argument("-m", "--method", choices=['ETKDG', 'ETKDGv3', 'MMFF'], 
                       default='ETKDG', help="Conformer generation method")
    parser.add_argument("--max-conformers", type=int, default=100,
                       help="Maximum conformers to dock per ligand (default: 100)")
    
    args = parser.parse_args()
    
    protein_file = Path(args.protein)
    ligands_input = Path(args.ligands)
    
    print("COMBINED CONFORMER GENERATION AND RIGID DOCKING")
    print("=" * 60)
    print(f"Protein: {protein_file}")
    print(f"Ligands: {ligands_input}")
    print(f"Conformers per ligand: {args.conformers}")
    print(f"Method: {args.method}")
    
    if not check_dependencies():
        return 1
    
    if not protein_file.exists():
        print(f"ERROR: Protein file not found: {protein_file}")
        return 1
    
    # Find ligand files
    if ligands_input.is_file():
        print(f"Processing single SDF file: {ligands_input}")
        suppl = Chem.SDMolSupplier(str(ligands_input), removeHs=False, sanitize=False)
        num_mols = len([m for m in suppl if m is not None])
        print(f"Found {num_mols} molecules in SDF file")
        ligand_files = [ligands_input]
        is_single_file = True
    else:
        ligand_files = find_ligand_files(ligands_input)
        if not ligand_files:
            print(f"ERROR: No SDF files found in {ligands_input}")
            return 1
        print(f"Found {len(ligand_files)} SDF files")
        is_single_file = False
    
    # Setup output directory
    output_dir = Path(f"conformer_docking_{protein_file.stem}")
    output_dir.mkdir(exist_ok=True)
    
    conformer_dir = output_dir / "conformers"
    conformer_dir.mkdir(exist_ok=True)
    
    results_dir = output_dir / "results"
    results_dir.mkdir(exist_ok=True)
    
    print(f"Output directory: {output_dir}")
    
    start_time = time.time()
    
    # Prepare protein
    print(f"\n1. PROTEIN PREPARATION")
    print("-" * 30)
    
    center, box_size = find_binding_site_from_ligand(protein_file)
    
    clean_protein = output_dir / "protein_clean.pdb"
    if not extract_protein_from_complex(protein_file, clean_protein):
        print("FAILED: Could not extract clean protein")
        return 1
    
    receptor_pdbqt = output_dir / "receptor.pdbqt"
    if not create_protein_pdbqt(clean_protein, receptor_pdbqt):
        print("FAILED: Could not create receptor PDBQT")
        return 1
    
    # Process ligands
    print(f"\n2. CONFORMER GENERATION AND DOCKING")
    print("-" * 40)
    
    all_results = []
    successful_ligands = 0
    failed_ligands = 0
    
    generator = ConformerGenerator()
    
    if is_single_file:
        # Process single SDF with multiple molecules
        suppl = Chem.SDMolSupplier(str(ligands_input), removeHs=False, sanitize=False)
        
        for mol_idx, mol in enumerate(suppl):
            if mol is None:
                continue
            
            if mol.HasProp("_Name"):
                ligand_name = mol.GetProp("_Name").strip()
            else:
                ligand_name = f"molecule_{mol_idx+1}"
            
            ligand_name = ligand_name.replace('/', '_').replace('\\', '_')
            
            print(f"\n{mol_idx+1}/{num_mols}: {ligand_name}")
            
            # Load molecule
            if not generator.load_molecule_from_mol_object(mol):
                print("  FAILED: Could not load molecule")
                failed_ligands += 1
                continue
            
            # Check molecule properties
            try:
                num_heavy_atoms = mol.GetNumHeavyAtoms()
                if num_heavy_atoms < 5 or num_heavy_atoms > 80:
                    print(f"  SKIPPED: {num_heavy_atoms} heavy atoms (outside 5-80 range)")
                    continue
            except:
                print("  SKIPPED: Could not analyze molecule")
                continue
            
            # Generate conformers
            print(f"  Generating {args.conformers} conformers...", end=" ")
            n_conformers = generator.generate_conformers(args.conformers, args.method)
            
            if n_conformers == 0:
                print("FAILED")
                failed_ligands += 1
                continue
            
            print(f"OK ({n_conformers} conformers)")
            
            # Dock conformers
            conformers = generator.get_conformers()
            max_to_dock = min(len(conformers), args.max_conformers)
            
            print(f"  Docking {max_to_dock} conformers...", end=" ")
            
            best_score = float('inf')
            best_conformer_id = None
            docked_count = 0
            
            for i, (conf_mol, conf_id, energy) in enumerate(conformers[:max_to_dock]):
                # Convert conformer to PDBQT
                temp_ligand_pdbqt = conformer_dir / f"{ligand_name}_conf_{conf_id}.pdbqt"
                
                if convert_conformer_to_pdbqt(conf_mol, temp_ligand_pdbqt):
                    # Dock conformer
                    temp_output_pdbqt = results_dir / f"{ligand_name}_conf_{conf_id}_docked.pdbqt"
                    
                    score, success = run_docking(receptor_pdbqt, temp_ligand_pdbqt, center, box_size, temp_output_pdbqt)
                    
                    if success and score is not None:
                        docked_count += 1
                        if score < best_score:
                            best_score = score
                            best_conformer_id = conf_id
                    
                    # Clean up temporary files
                    if temp_ligand_pdbqt.exists():
                        temp_ligand_pdbqt.unlink()
                    if not (success and score is not None) and temp_output_pdbqt.exists():
                        temp_output_pdbqt.unlink()
            
            if docked_count > 0:
                print(f"OK ({docked_count} successful, best: {best_score:.2f} kcal/mol)")
                all_results.append({
                    'name': ligand_name,
                    'score': best_score,
                    'conformer_id': best_conformer_id,
                    'total_conformers': n_conformers,
                    'docked_conformers': docked_count,
                    'success': True
                })
                successful_ligands += 1
            else:
                print("FAILED (no successful docking)")
                all_results.append({
                    'name': ligand_name,
                    'score': None,
                    'success': False
                })
                failed_ligands += 1
    
    else:
        # Process multiple SDF files
        for file_idx, ligand_file in enumerate(ligand_files, 1):
            ligand_name = ligand_file.stem
            print(f"\n{file_idx}/{len(ligand_files)}: {ligand_name}")
            
            # Load molecule
            if not generator.load_molecule(ligand_file, 0):
                print("  FAILED: Could not load molecule")
                failed_ligands += 1
                continue
            
            # Check molecule properties
            try:
                num_heavy_atoms = generator.molecule.GetNumHeavyAtoms()
                if num_heavy_atoms < 5 or num_heavy_atoms > 80:
                    print(f"  SKIPPED: {num_heavy_atoms} heavy atoms (outside 5-80 range)")
                    continue
            except:
                print("  SKIPPED: Could not analyze molecule")
                continue
            
            # Generate conformers
            print(f"  Generating {args.conformers} conformers...", end=" ")
            n_conformers = generator.generate_conformers(args.conformers, args.method)
            
            if n_conformers == 0:
                print("FAILED")
                failed_ligands += 1
                continue
            
            print(f"OK ({n_conformers} conformers)")
            
            # Dock conformers
            conformers = generator.get_conformers()
            max_to_dock = min(len(conformers), args.max_conformers)
            
            print(f"  Docking {max_to_dock} conformers...", end=" ")
            
            best_score = float('inf')
            best_conformer_id = None
            docked_count = 0
            
            for i, (conf_mol, conf_id, energy) in enumerate(conformers[:max_to_dock]):
                # Convert conformer to PDBQT
                temp_ligand_pdbqt = conformer_dir / f"{ligand_name}_conf_{conf_id}.pdbqt"
                
                if convert_conformer_to_pdbqt(conf_mol, temp_ligand_pdbqt):
                    # Dock conformer
                    temp_output_pdbqt = results_dir / f"{ligand_name}_conf_{conf_id}_docked.pdbqt"
                    
                    score, success = run_docking(receptor_pdbqt, temp_ligand_pdbqt, center, box_size, temp_output_pdbqt)
                    
                    if success and score is not None:
                        docked_count += 1
                        if score < best_score:
                            best_score = score
                            best_conformer_id = conf_id
                    
                    # Clean up temporary files
                    if temp_ligand_pdbqt.exists():
                        temp_ligand_pdbqt.unlink()
                    if not (success and score is not None) and temp_output_pdbqt.exists():
                        temp_output_pdbqt.unlink()
            
            if docked_count > 0:
                print(f"OK ({docked_count} successful, best: {best_score:.2f} kcal/mol)")
                all_results.append({
                    'name': ligand_name,
                    'score': best_score,
                    'conformer_id': best_conformer_id,
                    'total_conformers': n_conformers,
                    'docked_conformers': docked_count,
                    'success': True
                })
                successful_ligands += 1
            else:
                print("FAILED (no successful docking)")
                all_results.append({
                    'name': ligand_name,
                    'score': None,
                    'success': False
                })
                failed_ligands += 1
    
    # Results summary
    total_time = time.time() - start_time
    total_ligands = successful_ligands + failed_ligands
    
    print(f"\n3. RESULTS SUMMARY")
    print("-" * 30)
    print(f"Total time: {total_time:.1f}s ({total_time/60:.1f} minutes)")
    print(f"Successful ligands: {successful_ligands}/{total_ligands}")
    print(f"Failed ligands: {failed_ligands}")
    
    successful_results = [r for r in all_results if r['success']]
    successful_results.sort(key=lambda x: x['score'])
    
    if successful_results:
        print(f"\nTOP RIGID DOCKING RESULTS:")
        print("Rank  Ligand                  Score (kcal/mol)  Conformers  Best Conf")
        print("-" * 70)
        
        for i, result in enumerate(successful_results[:15], 1):
            name = result['name'][:18]
            score = result['score']
            n_conf = result['total_conformers']
            best_conf = result['conformer_id']
            print(f"{i:2d}.   {name:<18} {score:8.2f}        {n_conf:3d}       {best_conf:3d}")
        
        # Save results
        results_file = output_dir / "rigid_docking_results.txt"
        with open(results_file, 'w') as f:
            f.write("CONFORMER-BASED RIGID DOCKING RESULTS\n")
            f.write("=" * 45 + "\n\n")
            f.write(f"Protein: {protein_file}\n")
            f.write(f"Ligands: {ligands_input}\n")
            f.write(f"Conformers per ligand: {args.conformers}\n")
            f.write(f"Method: {args.method}\n")
            f.write(f"Box center: ({center[0]:.2f}, {center[1]:.2f}, {center[2]:.2f})\n")
            f.write(f"Box size: {box_size:.1f} Å\n")
            f.write(f"Successful: {successful_ligands}/{total_ligands}\n")
            f.write(f"Time: {total_time:.1f}s\n\n")
            
            f.write("RESULTS:\n")
            f.write("Ligand                    Score(kcal/mol)  Conformers  BestConf  DockedConf\n")
            f.write("-" * 75 + "\n")
            for result in successful_results:
                f.write(f"{result['name']:<25} {result['score']:8.2f}        "
                       f"{result['total_conformers']:3d}       {result['conformer_id']:3d}       "
                       f"{result['docked_conformers']:3d}\n")
        
        print(f"\nResults saved to: {results_file}")
        print(f"Best docked conformers saved in: {results_dir}")
        
    else:
        print("\n[X] No successful rigid dockings!")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
