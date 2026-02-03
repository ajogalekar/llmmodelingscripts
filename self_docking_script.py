#!/usr/bin/env python3
"""
UNIVERSAL PDB FILE DOCKING SCRIPT
=================================

Simple script that takes any PDB file and performs ligand docking.
Automatically detects ligands and sets up everything for you.

Usage:
    python dock_any_pdb.py structure.pdb
    python dock_any_pdb.py structure.pdb --ligand ATP --box_size 30
    python dock_any_pdb.py structure.pdb --center "10,15,8" --modes 20

Requirements:
    - conda install -c conda-forge autodock-vina openbabel
    - pip install biopython numpy

Author: Universal docking template
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path
import numpy as np
from Bio.PDB import PDBParser, PDBIO, Select

class LigandSelect(Select):
    def __init__(self, ligand_resname):
        self.ligand_resname = ligand_resname
    def accept_residue(self, residue):
        return residue.get_resname() == self.ligand_resname

class ProteinSelect(Select):
    def accept_residue(self, residue):
        standard_aa = ['ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY', 
                      'HIS', 'ILE', 'LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER', 
                      'THR', 'TRP', 'TYR', 'VAL']
        return residue.get_resname() in standard_aa

def analyze_pdb(pdb_file):
    """Analyze PDB file and show what's inside"""
    print(f"🔍 Analyzing {pdb_file}...")
    
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("structure", pdb_file)
    
    standard_residues = {
        'ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY', 
        'HIS', 'ILE', 'LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER', 
        'THR', 'TRP', 'TYR', 'VAL', 'HOH', 'WAT', 'H2O'
    }
    
    ligands = []
    protein_chains = []
    total_atoms = 0
    
    for model in structure:
        for chain in model:
            chain_residues = list(chain.get_residues())
            protein_residues = [r for r in chain_residues if r.get_resname() in standard_residues]
            
            if protein_residues:
                protein_chains.append({
                    'chain_id': chain.id,
                    'residues': len(protein_residues),
                    'atoms': sum(len(list(r.get_atoms())) for r in protein_residues)
                })
            
            for residue in chain_residues:
                resname = residue.get_resname()
                if resname not in standard_residues and resname not in ['HOH', 'WAT', 'H2O']:
                    atom_count = len(list(residue.get_atoms()))
                    ligands.append({
                        'resname': resname,
                        'chain': chain.id,
                        'resnum': residue.id[1],
                        'atoms': atom_count
                    })
                    total_atoms += atom_count
    
    # Display analysis
    print(f"📊 Structure Contents:")
    print(f"   🧬 Protein chains: {len(protein_chains)}")
    for chain_info in protein_chains:
        print(f"      Chain {chain_info['chain_id']}: {chain_info['residues']} residues, {chain_info['atoms']} atoms")
    
    print(f"   💊 Ligands found: {len(ligands)}")
    if ligands:
        for lig in sorted(ligands, key=lambda x: x['atoms'], reverse=True):
            print(f"      {lig['resname']:>6} (Chain {lig['chain']}, Res {lig['resnum']:>3}): {lig['atoms']:>3} atoms")
    else:
        print("      ❌ No ligands detected!")
        print("      💡 This PDB might only contain protein - you'll need to provide a separate ligand")
    
    return ligands, protein_chains

def extract_structures(pdb_file, ligand_resname, output_dir):
    """Extract protein and ligand to separate files"""
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("complex", pdb_file)
    io = PDBIO()
    
    output_dir = Path(output_dir)
    
    # Extract protein
    io.set_structure(structure)
    protein_file = output_dir / "protein.pdb"
    io.save(str(protein_file), ProteinSelect())
    
    # Extract ligand
    ligand_file = output_dir / "ligand.pdb"
    io.save(str(ligand_file), LigandSelect(ligand_resname))
    
    # Verify extraction
    protein_size = protein_file.stat().st_size if protein_file.exists() else 0
    ligand_size = ligand_file.stat().st_size if ligand_file.exists() else 0
    
    if protein_size == 0:
        raise Exception("❌ No protein found - check your PDB file")
    if ligand_size == 0:
        raise Exception(f"❌ Ligand '{ligand_resname}' not found in structure")
    
    print(f"✅ Extracted protein: {protein_file.name} ({protein_size} bytes)")
    print(f"✅ Extracted ligand: {ligand_file.name} ({ligand_size} bytes)")
    
    return protein_file, ligand_file

def get_ligand_center_and_size(ligand_file):
    """Calculate ligand center and suggest box size"""
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("ligand", ligand_file)
    
    coords = []
    for model in structure:
        for chain in model:
            for residue in chain:
                for atom in residue:
                    coords.append(atom.coord)
    
    if not coords:
        raise Exception("No atoms found in ligand")
    
    coords = np.array(coords)
    center = np.mean(coords, axis=0)
    
    # Calculate ligand dimensions
    distances = np.linalg.norm(coords - center, axis=1)
    max_distance = np.max(distances)
    ligand_span = max_distance * 2
    
    # Suggest box size (ligand + buffer)
    suggested_box = max(20.0, ligand_span + 8.0)
    suggested_box = min(suggested_box, 30.0)  # Cap at 30Å
    
    print(f"🎯 Ligand center: ({center[0]:.2f}, {center[1]:.2f}, {center[2]:.2f})")
    print(f"📏 Ligand span: {ligand_span:.1f} Å")
    print(f"💡 Suggested box size: {suggested_box:.1f} Å")
    
    return center, suggested_box

def convert_to_pdbqt(input_file, output_file, mol_type):
    """Convert PDB to PDBQT using Open Babel"""
    print(f"🔄 Converting {mol_type} to PDBQT...")
    
    try:
        if mol_type == "protein":
            cmd = ["obabel", str(input_file), "-O", str(output_file), 
                   "-xr", "-h", "--partialcharge", "gasteiger"]
        else:  # ligand
            cmd = ["obabel", str(input_file), "-O", str(output_file),
                   "-h", "--partialcharge", "gasteiger"]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0 and Path(output_file).exists() and Path(output_file).stat().st_size > 0:
            size = Path(output_file).stat().st_size
            print(f"✅ {mol_type} PDBQT created: {size} bytes")
            return True
        else:
            print(f"❌ {mol_type} conversion failed: {result.stderr}")
            return False
            
    except FileNotFoundError:
        print("❌ Open Babel not found. Install: conda install -c conda-forge openbabel")
        return False

def run_docking(receptor_pdbqt, ligand_pdbqt, center, box_size, num_modes, output_dir):
    """Run AutoDock Vina docking"""
    
    output_file = Path(output_dir) / "docked_poses.pdbqt"
    
    cmd = [
        "vina",
        "--receptor", str(receptor_pdbqt),
        "--ligand", str(ligand_pdbqt),
        "--out", str(output_file),
        "--center_x", f"{center[0]:.2f}",
        "--center_y", f"{center[1]:.2f}",
        "--center_z", f"{center[2]:.2f}",
        "--size_x", str(box_size),
        "--size_y", str(box_size),
        "--size_z", str(box_size),
        "--num_modes", str(num_modes),
        "--exhaustiveness", "8"
    ]
    
    print(f"🚀 Running AutoDock Vina...")
    print(f"   📍 Center: ({center[0]:.2f}, {center[1]:.2f}, {center[2]:.2f})")
    print(f"   📦 Box: {box_size} × {box_size} × {box_size} Å")
    print(f"   🎯 Modes: {num_modes}")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        
        # Save log
        log_file = Path(output_dir) / "vina_log.txt"
        with open(log_file, 'w') as f:
            f.write(" ".join(cmd) + "\n\n")
            f.write(f"Return code: {result.returncode}\n\n")
            f.write("STDOUT:\n" + result.stdout + "\n\n")
            f.write("STDERR:\n" + result.stderr + "\n")
        
        if result.returncode == 0:
            print("🎉 Docking completed successfully!")
            
            # Show results
            print("\n📊 Results:")
            for line in result.stdout.split('\n'):
                if ('mode' in line.lower() and 'affinity' in line.lower()) or \
                   '---' in line or \
                   (line.strip() and len(line.split()) >= 3 and line.split()[0].isdigit()):
                    print(f"   {line}")
            
            return output_file
        else:
            print(f"❌ Docking failed: {result.stderr}")
            return None
            
    except subprocess.TimeoutExpired:
        print("❌ Docking timed out")
        return None
    except FileNotFoundError:
        print("❌ Vina not found. Install: conda install -c conda-forge autodock-vina")
        return None

def extract_poses(docked_file, output_dir):
    """Extract poses as individual PDB files"""
    poses_dir = Path(output_dir) / "poses"
    poses_dir.mkdir(exist_ok=True)
    
    print(f"📦 Extracting poses...")
    
    pose_lines = []
    pose_num = 1
    
    with open(docked_file, 'r') as f:
        for line in f:
            if line.startswith('MODEL'):
                pose_lines = []
            elif line.startswith('ENDMDL'):
                if pose_lines:
                    pose_file = poses_dir / f"pose_{pose_num:02d}.pdb"
                    with open(pose_file, 'w') as pf:
                        for pose_line in pose_lines:
                            if pose_line.startswith(('ATOM', 'HETATM')):
                                clean_line = pose_line[:66].rstrip() + "\n"
                                pf.write(clean_line)
                    print(f"   ✅ {pose_file.name}")
                    pose_num += 1
            else:
                pose_lines.append(line)
    
    return pose_num - 1

def main():
    parser = argparse.ArgumentParser(
        description='Universal PDB docking script - works with any PDB file!',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python dock_any_pdb.py my_protein.pdb                    # Auto-detect everything
  python dock_any_pdb.py complex.pdb --ligand ATP          # Specify ligand
  python dock_any_pdb.py structure.pdb --box_size 30       # Custom box size
  python dock_any_pdb.py protein.pdb --center "10,15,8"    # Custom center
        """
    )
    
    parser.add_argument('pdb_file', help='PDB file to dock')
    parser.add_argument('--ligand', help='Ligand residue name (auto-select largest if not specified)')
    parser.add_argument('--center', help='Binding site center as "x,y,z" (auto-calculate if not specified)')
    parser.add_argument('--box_size', type=float, help='Search box size in Å (auto-calculate if not specified)')
    parser.add_argument('--modes', type=int, default=10, help='Number of docking modes (default: 10)')
    parser.add_argument('--output', help='Output directory (default: {pdb_filename}_docking)')
    
    args = parser.parse_args()
    
    # Validate input file
    pdb_file = Path(args.pdb_file)
    if not pdb_file.exists():
        print(f"❌ PDB file not found: {pdb_file}")
        return 1
    
    # Set output directory
    if args.output:
        output_dir = Path(args.output)
    else:
        output_dir = Path(f"{pdb_file.stem}_docking")
    
    output_dir.mkdir(exist_ok=True)
    
    print("🧬 UNIVERSAL PDB DOCKING SCRIPT")
    print("=" * 50)
    print(f"📂 Input: {pdb_file}")
    print(f"📁 Output: {output_dir}")
    
    try:
        # Step 1: Analyze the PDB file
        ligands, protein_chains = analyze_pdb(pdb_file)
        
        if not protein_chains:
            print("❌ No protein found in PDB file!")
            return 1
        
        if not ligands:
            print("❌ No ligands found in PDB file!")
            print("💡 You need a PDB with both protein and ligand for docking")
            return 1
        
        # Step 2: Select ligand
        if args.ligand:
            target_ligand = args.ligand.upper()
            if not any(lig['resname'] == target_ligand for lig in ligands):
                available = [lig['resname'] for lig in ligands]
                print(f"❌ Ligand '{target_ligand}' not found. Available: {available}")
                return 1
        else:
            # Auto-select largest ligand
            target_ligand = max(ligands, key=lambda x: x['atoms'])['resname']
        
        print(f"🎯 Using ligand: {target_ligand}")
        
        # Step 3: Extract structures
        protein_file, ligand_file = extract_structures(pdb_file, target_ligand, output_dir)
        
        # Step 4: Determine binding site
        if args.center:
            try:
                center_coords = [float(x.strip()) for x in args.center.split(',')]
                if len(center_coords) != 3:
                    raise ValueError
                center = np.array(center_coords)
                print(f"🎯 Using specified center: ({center[0]:.2f}, {center[1]:.2f}, {center[2]:.2f})")
            except:
                print("❌ Invalid center format. Use: 'x,y,z' (e.g., '10.5,15.2,8.8')")
                return 1
        else:
            center, suggested_box = get_ligand_center_and_size(ligand_file)
            if not args.box_size:
                args.box_size = suggested_box
        
        # Use default box size if not specified
        if not args.box_size:
            args.box_size = 25.0
        
        print(f"📦 Box size: {args.box_size} Å")
        
        # Step 5: Convert to PDBQT
        receptor_pdbqt = output_dir / "receptor.pdbqt"
        ligand_pdbqt = output_dir / "ligand.pdbqt"
        
        if not convert_to_pdbqt(protein_file, receptor_pdbqt, "protein"):
            return 1
        if not convert_to_pdbqt(ligand_file, ligand_pdbqt, "ligand"):
            return 1
        
        # Step 6: Run docking
        docked_file = run_docking(receptor_pdbqt, ligand_pdbqt, center, 
                                 args.box_size, args.modes, output_dir)
        
        if not docked_file:
            return 1
        
        # Step 7: Extract poses
        num_poses = extract_poses(docked_file, output_dir)
        
        print(f"\n🎉 SUCCESS!")
        print(f"📁 Results in: {output_dir}/")
        print(f"🎯 Generated: {num_poses} poses")
        print(f"📄 Individual poses: {output_dir}/poses/pose_XX.pdb")
        print(f"📝 Log file: {output_dir}/vina_log.txt")
        
        return 0
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
