#!/usr/bin/env python3
"""
UNIVERSAL PROTEIN-LIGAND DOCKING SCRIPT - IMPROVED
==================================================

Robust docking script using RDKit for ligand preparation and AutoDock Vina.
Works for any protein target with automatic binding site detection from existing ligand.
Improved timeout handling and molecular preparation for large/complex ligands.

Usage:
    python universal_docking.py protein.pdb ligands.sdf
    python universal_docking.py protein.pdb ligands_directory/

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

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem, Descriptors, SaltRemover
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

def is_molecule_suitable_for_docking(mol):
    """Check if molecule is suitable for docking (size, complexity filters)"""
    if mol is None:
        return False, "Invalid molecule"
    
    try:
        num_heavy_atoms = mol.GetNumHeavyAtoms()
        
        # Size filters
        if num_heavy_atoms < 5:
            return False, f"Too small: {num_heavy_atoms} heavy atoms"
        
        if num_heavy_atoms > 80:  # Reduced from 100 for better performance
            return False, f"Too large: {num_heavy_atoms} heavy atoms (>80)"
        
        # Molecular weight filter
        mw = Descriptors.MolWt(mol)
        if mw > 800:  # Saquinavir is 670.8, so this catches it
            return False, f"Molecular weight too high: {mw:.1f} Da (>800)"
        
        # Rotatable bonds filter (flexibility)
        rotatable_bonds = Descriptors.NumRotatableBonds(mol)
        if rotatable_bonds > 15:
            return False, f"Too flexible: {rotatable_bonds} rotatable bonds (>15)"
        
        # Ring complexity filter
        ring_info = mol.GetRingInfo()
        num_rings = ring_info.NumRings()
        if num_rings > 8:
            return False, f"Too complex: {num_rings} rings (>8)"
        
        return True, "Suitable"
        
    except Exception as e:
        return False, f"Error analyzing molecule: {e}"

def cleanup_ligand_molecule(mol):
    """
    Robust ligand cleanup protocol with timeout protection:
    1. Remove salts and counterions
    2. Keep only largest fragment
    3. Sanitize molecule
    4. Add hydrogens
    5. Generate proper 3D conformer (with timeout)
    """
    
    if mol is None:
        return None
    
    try:
        # Step 1: Try to sanitize first (this catches many issues early)
        try:
            Chem.SanitizeMol(mol, catchErrors=True)
        except:
            return None
        
        # Step 2: Remove common salts (Cl-, Br-, Na+, etc.)
        try:
            remover = SaltRemover.SaltRemover()
            mol = remover.StripMol(mol)
        except:
            pass
        
        # Step 3: If multiple fragments remain, keep the largest
        try:
            frags = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=False)
            if len(frags) > 1:
                mol = max(frags, key=lambda m: m.GetNumAtoms())
        except:
            pass
        
        # Step 4: Sanitize again after cleanup
        try:
            Chem.SanitizeMol(mol)
        except:
            return None
        
        # Step 5: Handle hydrogens carefully
        try:
            mol = Chem.RemoveHs(mol, sanitize=False)
            for atom in mol.GetAtoms():
                atom.UpdatePropertyCache(strict=False)
            mol = Chem.AddHs(mol, addCoords=False)
        except Exception as e:
            return None
        
        # Step 6: Generate 3D conformer with timeout protection
        try:
            # Set up signal handler for timeout
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(60)  # 60 second timeout for conformer generation
            
            # Try ETKDG first (best quality but slower)
            params = AllChem.ETKDGv3()
            params.randomSeed = 42
            params.useRandomCoords = False
            params.numThreads = 1  # Single thread for better timeout control
            params.maxAttempts = 3  # Reduced attempts for large molecules
            
            result = AllChem.EmbedMolecule(mol, params)
            
            if result == -1:
                # Try with random coordinates
                params.useRandomCoords = True
                params.maxAttempts = 1  # Single attempt
                result = AllChem.EmbedMolecule(mol, params)
                
                if result == -1:
                    # Last resort: basic embedding
                    result = AllChem.EmbedMolecule(mol, randomSeed=42, maxAttempts=1)
                    if result == -1:
                        signal.alarm(0)  # Cancel alarm
                        return None
            
            # Cancel the alarm
            signal.alarm(0)
            
        except TimeoutException:
            print(" [TIMEOUT during conformer generation]", end="")
            return None
        except Exception:
            signal.alarm(0)  # Make sure to cancel alarm
            return None
        
        # Step 7: Optimize geometry with timeout protection (optional, can be skipped for large molecules)
        try:
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(30)  # 30 second timeout for optimization
            
            props = AllChem.MMFFGetMoleculeProperties(mol)
            if props is not None:
                AllChem.MMFFOptimizeMolecule(mol, maxIters=200, mmffVariant='MMFF94')  # Reduced iterations
            else:
                AllChem.UFFOptimizeMolecule(mol, maxIters=200)  # Reduced iterations
            
            signal.alarm(0)  # Cancel alarm
            
        except TimeoutException:
            # Optimization timeout is not fatal, continue with unoptimized structure
            pass
        except:
            signal.alarm(0)  # Make sure to cancel alarm
            pass
        
        return mol
        
    except Exception as e:
        return None

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

def prepare_ligand_rdkit(sdf_file, ligand_index, pdbqt_file, use_meeko=True):
    """Prepare ligand using RDKit with robust cleanup and proper PDBQT generation"""
    
    try:
        suppl = Chem.SDMolSupplier(str(sdf_file), removeHs=False, sanitize=False)
        
        mol = None
        for idx, m in enumerate(suppl):
            if idx == ligand_index:
                mol = m
                break
        
        if mol is None:
            print("FAILED (invalid SDF or index)")
            return False
        
        # Clean up the molecule FIRST - this handles sanitization
        mol = cleanup_ligand_molecule(mol)
        
        if mol is None:
            print("FAILED (cleanup failed)")
            return False
        
        # NOW check if molecule is suitable for docking (after sanitization)
        suitable, reason = is_molecule_suitable_for_docking(mol)
        if not suitable:
            print(f"SKIPPED ({reason})")
            return False
        
        num_atoms = mol.GetNumAtoms()
        num_heavy_atoms = mol.GetNumHeavyAtoms()
        
        # Try Meeko first (with timeout)
        if use_meeko and MEEKO_AVAILABLE:
            try:
                # Set up timeout for Meeko
                signal.signal(signal.SIGALRM, timeout_handler)
                signal.alarm(120)  # 2 minute timeout for Meeko
                
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
                    
                    signal.alarm(0)  # Cancel alarm
                    print(f"OK (meeko, {num_heavy_atoms} atoms)")
                    return True
                
                signal.alarm(0)  # Cancel alarm
                
            except TimeoutException:
                signal.alarm(0)
                print("TIMEOUT (meeko) - trying fallback...", end=" ")
                # Continue to fallback method
            except Exception as e:
                signal.alarm(0)
                # Continue to fallback method
        
        # Fallback: Use temporary PDB then obabel conversion
        try:
            temp_pdb = pdbqt_file.with_suffix('.temp.pdb')
            Chem.MolToPDBFile(mol, str(temp_pdb))
            
            cmd = ["obabel", str(temp_pdb), "-O", str(pdbqt_file), "-xh"]
            result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=60)
            
            if temp_pdb.exists():
                temp_pdb.unlink()
            
            if result.returncode == 0 and pdbqt_file.exists():
                print(f"OK (rdkit+obabel, {num_heavy_atoms} atoms)")
                return True
            else:
                print("FAILED (conversion error)")
                return False
                
        except subprocess.TimeoutExpired:
            print("FAILED (obabel timeout)")
            if temp_pdb.exists():
                temp_pdb.unlink()
            return False
        except Exception as e:
            print(f"FAILED ({e})")
            return False
        
    except Exception as e:
        print(f"FAILED ({e})")
        return False

def run_docking(receptor_pdbqt, ligand_pdbqt, center, box_size, ligand_name, output_dir):
    """Run docking with optimized parameters and adaptive settings"""
    
    temp_output = output_dir / f"{ligand_name}_docked.pdbqt"
    
    # Adaptive parameters based on box size (larger molecules need longer search)
    if box_size > 25:
        exhaustiveness = 8  # Reduced for large binding sites
        timeout = 600  # 10 minutes for large molecules
    else:
        exhaustiveness = 12
        timeout = 300  # 5 minutes for regular molecules
    
    cmd = [
        "vina",
        "--receptor", str(receptor_pdbqt),
        "--ligand", str(ligand_pdbqt),
        "--out", str(temp_output),
        "--center_x", f"{center[0]:.3f}",
        "--center_y", f"{center[1]:.3f}",
        "--center_z", f"{center[2]:.3f}",
        "--size_x", str(box_size),
        "--size_y", str(box_size),
        "--size_z", str(box_size),
        "--num_modes", "3",
        "--exhaustiveness", str(exhaustiveness),
        "--energy_range", "3"
    ]
    
    print(f"    Docking...", end=" ")
    
    try:
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=timeout)
        
        if result.returncode == 0 and temp_output.exists():
            
            best_score = None
            poses_found = 0
            
            for line in result.stdout.split('\n'):
                if line.strip() and line.strip()[0].isdigit():
                    parts = line.split()
                    if len(parts) >= 2:
                        try:
                            pose_num = int(parts[0])
                            score = float(parts[1])
                            poses_found += 1
                            
                            if pose_num == 1:
                                best_score = score
                        except (ValueError, IndexError):
                            pass
            
            print(f"OK ({best_score:.2f} kcal/mol, {poses_found} poses)")
            
            return {
                'name': ligand_name,
                'score': best_score,
                'poses': poses_found,
                'success': best_score is not None,
                'pdbqt_output': temp_output
            }
        
        else:
            print("FAILED")
            if result.stderr:
                error_msg = result.stderr[:200]
                print(f"      Error: {error_msg}")
            
            return {
                'name': ligand_name,
                'score': None,
                'success': False,
                'error': result.stderr[:200] if result.stderr else 'Unknown error'
            }
        
    except subprocess.TimeoutExpired:
        print(f"TIMEOUT (>{timeout//60} min)")
        if temp_output.exists():
            temp_output.unlink()
        
        return {
            'name': ligand_name,
            'score': None,
            'success': False,
            'error': f'Timeout (>{timeout//60} min)'
        }

def main():
    if len(sys.argv) != 3:
        print("Usage: python universal_docking.py protein.pdb ligands.sdf")
        print("       python universal_docking.py protein.pdb ligands_directory/")
        return 1
    
    protein_file = Path(sys.argv[1])
    ligands_input = Path(sys.argv[2])
    
    print("UNIVERSAL PROTEIN-LIGAND DOCKING - IMPROVED")
    print("=" * 50)
    
    if not check_dependencies():
        return 1
    
    if not protein_file.exists():
        print(f"ERROR: Protein file not found: {protein_file}")
        return 1
    
    if ligands_input.is_file():
        print(f"Processing single SDF file: {ligands_input}")
        suppl = Chem.SDMolSupplier(str(ligands_input), removeHs=False, sanitize=False)
        num_mols = len([m for m in suppl if m is not None])
        print(f"Found {num_mols} molecules in SDF file")
        ligand_source = ligands_input
        is_single_file = True
    else:
        ligand_files = find_ligand_files(ligands_input)
        if not ligand_files:
            print(f"ERROR: No SDF files found in {ligands_input}")
            return 1
        print(f"Found {len(ligand_files)} SDF files")
        ligand_source = ligands_input
        is_single_file = False
    
    output_dir = Path(f"docking_{protein_file.stem}")
    output_dir.mkdir(exist_ok=True)
    
    results_dir = output_dir / "results"
    results_dir.mkdir(exist_ok=True)
    
    print(f"Output directory: {output_dir}")
    
    start_time = time.time()
    
    # Prepare protein
    print(f"\n1. PROTEIN PREPARATION")
    print("-" * 30)
    
    # Use original protein file (with ligand) to find binding site
    center, box_size = find_binding_site_from_ligand(protein_file)
    
    clean_protein = output_dir / "protein_clean.pdb"
    if not extract_protein_from_complex(protein_file, clean_protein):
        print("FAILED: Could not extract clean protein")
        return 1
    
    receptor_pdbqt = output_dir / "receptor.pdbqt"
    if not create_protein_pdbqt(clean_protein, receptor_pdbqt):
        print("FAILED: Could not create receptor PDBQT")
        return 1
    
    # Dock ligands
    print(f"\n2. LIGAND DOCKING")
    print("-" * 30)
    
    results = []
    successful = 0
    failed_conversion = 0
    failed_docking = 0
    skipped_large = 0
    
    if is_single_file:
        suppl = Chem.SDMolSupplier(str(ligand_source), removeHs=False, sanitize=False)
        
        for idx, mol in enumerate(suppl):
            if mol is None:
                continue
                
            if mol.HasProp("_Name"):
                ligand_name = mol.GetProp("_Name").strip()
            else:
                ligand_name = f"molecule_{idx+1}"
            
            ligand_name = ligand_name.replace('/', '_').replace('\\', '_')
            
            print(f"\n{idx+1}/{num_mols}: {ligand_name}")
            print(f"    Converting...", end=" ")
            
            ligand_pdbqt = results_dir / f"{ligand_name}.pdbqt"
            
            prep_result = prepare_ligand_rdkit(ligand_source, idx, ligand_pdbqt)
            if prep_result is False:
                # Check if it was skipped due to size/complexity
                if "SKIPPED" in str(prep_result):
                    skipped_large += 1
                else:
                    failed_conversion += 1
                results.append({'name': ligand_name, 'score': None, 'success': False, 'error': 'conversion_failed'})
                continue
            
            result = run_docking(receptor_pdbqt, ligand_pdbqt, center, 
                                    box_size, ligand_name, results_dir)
            results.append(result)
            
            if result['success']:
                successful += 1
            else:
                failed_docking += 1
            
            if ligand_pdbqt.exists() and not result.get('success', False):
                ligand_pdbqt.unlink()
    else:
        for i, ligand_file in enumerate(ligand_files, 1):
            ligand_name = ligand_file.stem
            print(f"\n{i:2d}/{len(ligand_files)}: {ligand_name}")
            print(f"    Converting {ligand_file.name}...", end=" ")
            
            ligand_pdbqt = results_dir / f"{ligand_name}.pdbqt"
            
            if not prepare_ligand_rdkit(ligand_file, 0, ligand_pdbqt):
                failed_conversion += 1
                results.append({'name': ligand_name, 'score': None, 'success': False, 'error': 'conversion_failed'})
                continue
            
            result = run_docking(receptor_pdbqt, ligand_pdbqt, center, 
                                    box_size, ligand_name, results_dir)
            results.append(result)
            
            if result['success']:
                successful += 1
            else:
                failed_docking += 1
            
            if ligand_pdbqt.exists() and not result.get('success', False):
                ligand_pdbqt.unlink()
    
    # Results summary
    total_time = time.time() - start_time
    total_ligands = num_mols if is_single_file else len(ligand_files)
    
    print(f"\n3. RESULTS SUMMARY")
    print("-" * 30)
    print(f"Total time: {total_time:.1f}s ({total_time/60:.1f} minutes)")
    print(f"Successful: {successful}/{total_ligands}")
    print(f"Failed conversions: {failed_conversion}")
    print(f"Failed docking: {failed_docking}")
    if skipped_large > 0:
        print(f"Skipped (too large/complex): {skipped_large}")
    
    successful_results = [r for r in results if r['success'] and r['score'] is not None]
    successful_results.sort(key=lambda x: x['score'])
    
    if successful_results:
        print(f"\nTOP DOCKING RESULTS:")
        print("Rank  Ligand                  Score (kcal/mol)  Poses")
        print("-" * 55)
        
        for i, result in enumerate(successful_results[:15], 1):
            name = result['name'][:20]
            score = result['score']
            poses = result.get('poses', '?')
            print(f"{i:2d}.   {name:<20} {score:8.2f}        {poses:3}")
        
        results_file = output_dir / "docking_results.txt"
        with open(results_file, 'w') as f:
            f.write("PROTEIN-LIGAND DOCKING RESULTS\n")
            f.write("=" * 40 + "\n\n")
            f.write(f"Protein: {protein_file}\n")
            f.write(f"Ligands: {ligand_source}\n")
            f.write(f"Binding site from ligand in structure\n")
            f.write(f"Box center: ({center[0]:.2f}, {center[1]:.2f}, {center[2]:.2f})\n")
            f.write(f"Box size: {box_size:.1f} Å\n")
            f.write(f"Successful: {successful}/{total_ligands}\n")
            f.write(f"Time: {total_time:.1f}s\n\n")
            
            f.write("RESULTS:\n")
            for result in successful_results:
                f.write(f"{result['name']:<25} {result['score']:8.2f} kcal/mol\n")
        
        print(f"\nResults saved to: {results_file}")
        
    else:
        print("\n[X] No successful dockings!")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
