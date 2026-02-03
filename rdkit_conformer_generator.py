#!/usr/bin/env python3
"""
Simple RDKit-based conformer generation
Reliable, fast, and straightforward approach
"""

import sys
import argparse
import numpy as np
from pathlib import Path

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem, rdMolAlign, rdMolDescriptors
    print("✓ RDKit available")
except ImportError:
    print("✗ RDKit not found. Install with: conda install -c conda-forge rdkit")
    sys.exit(1)

class RDKitConformerGenerator:
    """Simple RDKit conformer generation and analysis"""
    
    def __init__(self):
        self.molecule = None
        self.conformer_energies = []
        
    def load_molecule(self, input_file):
        """Load molecule from file"""
        print(f"Loading molecule from {input_file}...")
        
        input_path = Path(input_file)
        
        # Load based on file extension
        if input_path.suffix.lower() == '.pdb':
            mol = Chem.MolFromPDBFile(str(input_file), removeHs=False)
            if mol is None:
                mol = Chem.MolFromPDBFile(str(input_file), removeHs=True)
                if mol is not None:
                    mol = Chem.AddHs(mol)
        elif input_path.suffix.lower() == '.sdf':
            mol = Chem.MolFromMolFile(str(input_file), removeHs=False)
        elif input_path.suffix.lower() == '.mol2':
            mol = Chem.MolFromMol2File(str(input_file), removeHs=False)
        elif input_path.suffix.lower() == '.smiles' or input_path.suffix.lower() == '.smi':
            with open(input_file, 'r') as f:
                smiles = f.read().strip().split()[0]
            mol = Chem.MolFromSmiles(smiles)
            mol = Chem.AddHs(mol)
        else:
            raise ValueError(f"Unsupported file format: {input_path.suffix}")
        
        if mol is None:
            raise ValueError("Could not load molecule - check file format")
        
        # Sanitize molecule
        try:
            Chem.SanitizeMol(mol)
        except:
            raise ValueError("Could not sanitize molecule - check structure")
        
        self.molecule = mol
        
        # Print molecule info
        print(f"   Formula: {rdMolDescriptors.CalcMolFormula(mol)}")
        print(f"   Atoms: {mol.GetNumAtoms()}")
        print(f"   Heavy atoms: {mol.GetNumHeavyAtoms()}")
        print(f"   Rotatable bonds: {rdMolDescriptors.CalcNumRotatableBonds(mol)}")
        print(f"   Molecular weight: {rdMolDescriptors.CalcExactMolWt(mol):.1f}")
        
        return mol
    
    def generate_conformers(self, n_conformers=1000, method='ETKDG'):
        """Generate conformers using RDKit"""
        print(f"\nGenerating {n_conformers} conformers using {method}...")
        
        # Clear existing conformers
        self.molecule.RemoveAllConformers()
        
        # Generate conformers using method-specific approach
        if method == 'ETKDG':
            # Enhanced version with torsion knowledge
            conf_ids = AllChem.EmbedMultipleConfs(
                self.molecule,
                numConfs=n_conformers,
                randomSeed=42,
                useExpTorsionAnglePrefs=True,
                useBasicKnowledge=True,
                enforceChirality=True
            )
            
        elif method == 'ETKDGv3':
            # Use ETKDGv3 parameters
            try:
                params = AllChem.ETKDGv3()
                params.randomSeed = 42
                conf_ids = AllChem.EmbedMultipleConfs(
                    self.molecule,
                    numConfs=n_conformers,
                    params=params
                )
            except AttributeError:
                print("   ETKDGv3 not available, falling back to ETKDG")
                conf_ids = AllChem.EmbedMultipleConfs(
                    self.molecule,
                    numConfs=n_conformers,
                    randomSeed=42,
                    useExpTorsionAnglePrefs=True,
                    useBasicKnowledge=True,
                    enforceChirality=True
                )
            
        elif method == 'MMFF':
            # Basic distance geometry
            conf_ids = AllChem.EmbedMultipleConfs(
                self.molecule,
                numConfs=n_conformers,
                randomSeed=42
            )
            
        else:
            raise ValueError(f"Unknown method: {method}")
        
        if len(conf_ids) == 0:
            raise ValueError("No conformers could be generated. Check your structure.")
        
        print(f"   Generated {len(conf_ids)} initial conformers")
        
        # Optimize with force field
        print("   Optimizing with MMFF94...")
        
        # Try bulk optimization first (newer RDKit versions)
        try:
            results = AllChem.MMFFOptimizeMolConfs(
                self.molecule,
                maxIters=1000
            )
        except AttributeError:
            # Fallback for older RDKit versions - optimize individually
            print("   Using individual conformer optimization...")
            results = []
            for conf_id in conf_ids:
                try:
                    # Setup MMFF for this conformer
                    mp = AllChem.MMFFGetMoleculeProperties(self.molecule, mmffVariant='MMFF94')
                    if mp is None:
                        results.append((1, float('inf')))  # Failed
                        continue
                    
                    ff = AllChem.MMFFGetMoleculeForceField(self.molecule, mp, confId=conf_id)
                    if ff is None:
                        results.append((1, float('inf')))  # Failed
                        continue
                    
                    # Optimize
                    converged = ff.Minimize(maxIts=1000)
                    energy = ff.CalcEnergy()
                    results.append((converged, energy))
                    
                except Exception:
                    results.append((1, float('inf')))  # Failed
        
        # Filter successful optimizations
        successful_conformers = []
        self.conformer_energies = []
        
        for i, (converged, energy) in enumerate(results):
            if converged == 0 and energy != float('inf'):  # Successful convergence
                successful_conformers.append(conf_ids[i])
                self.conformer_energies.append(energy)
        
        print(f"   ✓ {len(successful_conformers)} conformers successfully optimized")
        
        if len(successful_conformers) == 0:
            raise ValueError("No conformers successfully optimized")
        
        # Remove failed conformers
        conf_ids_to_remove = set(conf_ids) - set(successful_conformers)
        for conf_id in sorted(conf_ids_to_remove, reverse=True):
            self.molecule.RemoveConformer(conf_id)
        
        # Print energy statistics
        min_energy = min(self.conformer_energies)
        max_energy = max(self.conformer_energies)
        mean_energy = np.mean(self.conformer_energies)
        
        print(f"   Energy range: {min_energy:.1f} to {max_energy:.1f} kcal/mol")
        print(f"   Energy span: {max_energy - min_energy:.1f} kcal/mol")
        print(f"   Mean energy: {mean_energy:.1f} kcal/mol")
        
        return len(successful_conformers)
    
    def cluster_conformers(self, rmsd_threshold=0.5):
        """Cluster conformers by RMSD"""
        print(f"\nClustering conformers (RMSD threshold: {rmsd_threshold} Å)...")
        
        n_conformers = self.molecule.GetNumConformers()
        if n_conformers < 2:
            print("   Only one conformer - no clustering needed")
            return list(range(n_conformers))
        
        # Calculate RMSD matrix
        print("   Calculating RMSD matrix...")
        rmsd_matrix = np.zeros((n_conformers, n_conformers))
        
        for i in range(n_conformers):
            for j in range(i + 1, n_conformers):
                rmsd = rdMolAlign.CalcRMS(self.molecule, self.molecule, i, j)
                rmsd_matrix[i, j] = rmsd
                rmsd_matrix[j, i] = rmsd
        
        # Greedy clustering - select diverse conformers
        selected = []
        energy_order = np.argsort(self.conformer_energies)
        
        # Always include lowest energy
        selected.append(energy_order[0])
        
        for conf_idx in energy_order[1:]:
            # Check if this conformer is similar to any selected ones
            is_unique = True
            for selected_idx in selected:
                if rmsd_matrix[conf_idx, selected_idx] < rmsd_threshold:
                    is_unique = False
                    break
            
            if is_unique:
                selected.append(conf_idx)
        
        print(f"   ✓ Selected {len(selected)} diverse conformers")
        
        return selected
    
    def analyze_conformers(self):
        """Analyze conformer ensemble"""
        print("\nConformer Analysis:")
        print("-" * 30)
        
        n_conformers = self.molecule.GetNumConformers()
        
        # Energy analysis
        min_energy = min(self.conformer_energies)
        relative_energies = [e - min_energy for e in self.conformer_energies]
        
        # Count conformers in energy ranges
        within_1 = sum(1 for e in relative_energies if e <= 1.0)
        within_2 = sum(1 for e in relative_energies if e <= 2.0)
        within_5 = sum(1 for e in relative_energies if e <= 5.0)
        within_10 = sum(1 for e in relative_energies if e <= 10.0)
        
        print(f"Total conformers: {n_conformers}")
        print(f"Within 1 kcal/mol: {within_1} ({within_1/n_conformers*100:.1f}%)")
        print(f"Within 2 kcal/mol: {within_2} ({within_2/n_conformers*100:.1f}%)")
        print(f"Within 5 kcal/mol: {within_5} ({within_5/n_conformers*100:.1f}%)")
        print(f"Within 10 kcal/mol: {within_10} ({within_10/n_conformers*100:.1f}%)")
        
        # RMSD analysis
        if n_conformers > 1:
            print("\nRMSD Analysis:")
            all_rmsds = []
            for i in range(n_conformers):
                for j in range(i + 1, n_conformers):
                    rmsd = rdMolAlign.CalcRMS(self.molecule, self.molecule, i, j)
                    all_rmsds.append(rmsd)
            
            print(f"RMSD range: {min(all_rmsds):.2f} to {max(all_rmsds):.2f} Å")
            print(f"Mean RMSD: {np.mean(all_rmsds):.2f} Å")
    
    def save_conformers(self, output_file, selected_indices=None, sort_by_energy=True):
        """Save conformers to SDF file"""
        print(f"\nSaving conformers to {output_file}...")
        
        if selected_indices is None:
            selected_indices = list(range(self.molecule.GetNumConformers()))
        
        # Sort by energy if requested
        if sort_by_energy:
            energy_data = [(self.conformer_energies[i], i) for i in selected_indices]
            energy_data.sort()
            selected_indices = [idx for _, idx in energy_data]
        
        writer = Chem.SDWriter(str(output_file))
        
        for rank, conf_id in enumerate(selected_indices):
            # Create copy with single conformer
            mol_copy = Chem.Mol(self.molecule)
            mol_copy.RemoveAllConformers()
            mol_copy.AddConformer(self.molecule.GetConformer(conf_id))
            
            # Add properties
            energy = self.conformer_energies[conf_id]
            min_energy = min(self.conformer_energies[i] for i in selected_indices)
            
            mol_copy.SetProp("Energy_kcal_mol", f"{energy:.2f}")
            mol_copy.SetProp("RelativeEnergy_kcal_mol", f"{energy - min_energy:.2f}")
            mol_copy.SetProp("ConformerID", str(conf_id))
            mol_copy.SetProp("Rank", str(rank + 1))
            
            writer.write(mol_copy)
        
        writer.close()
        print(f"   ✓ Saved {len(selected_indices)} conformers")
        
        # Save lowest energy as separate PDB
        if selected_indices:
            lowest_idx = selected_indices[0]
            pdb_file = Path(output_file).with_suffix('.pdb')
            
            mol_copy = Chem.Mol(self.molecule)
            mol_copy.RemoveAllConformers()
            mol_copy.AddConformer(self.molecule.GetConformer(lowest_idx))
            
            Chem.MolToPDBFile(mol_copy, str(pdb_file))
            print(f"   ✓ Lowest energy conformer saved as: {pdb_file}")

def main():
    parser = argparse.ArgumentParser(description="Generate conformers with RDKit")
    parser.add_argument("input", help="Input molecule file")
    parser.add_argument("-n", "--conformers", type=int, default=10, 
                       help="Number of conformers to generate")
    parser.add_argument("-o", "--output", default="conformers.sdf", 
                       help="Output SDF file")
    parser.add_argument("-m", "--method", choices=['ETKDG', 'ETKDGv3', 'MMFF'], 
                       default='ETKDG', help="Conformer generation method")
    parser.add_argument("-r", "--rmsd", type=float, default=0.5,
                       help="RMSD threshold for clustering (Å)")
    parser.add_argument("--cluster", action="store_true",
                       help="Cluster conformers by RMSD")
    
    args = parser.parse_args()
    
    print("RDKIT CONFORMER GENERATION")
    print("=" * 40)
    print(f"Input: {args.input}")
    print(f"Target conformers: {args.conformers}")
    print(f"Method: {args.method}")
    print(f"Output: {args.output}")
    
    try:
        # Initialize generator
        generator = RDKitConformerGenerator()
        
        # Load molecule
        generator.load_molecule(args.input)
        
        # Generate conformers
        n_generated = generator.generate_conformers(
            n_conformers=args.conformers,
            method=args.method
        )
        
        # Analyze conformers
        generator.analyze_conformers()
        
        # Cluster if requested
        if args.cluster:
            selected_indices = generator.cluster_conformers(args.rmsd)
        else:
            selected_indices = None
        
        # Save results
        generator.save_conformers(args.output, selected_indices)
        
        print(f"\n✓ Conformer generation completed!")
        if selected_indices:
            print(f"Generated {n_generated} conformers, saved {len(selected_indices)} diverse ones")
        else:
            print(f"Generated and saved {n_generated} conformers")
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
