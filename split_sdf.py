#!/usr/bin/env python3
"""
SDF FILE SPLITTER
=================

Splits a multi-compound SDF file into individual SDF files,
one per compound. Perfect for ChEMBL downloads!

Usage:
    python split_sdf.py input_file.sdf
    python split_sdf.py chembl_compounds.sdf
    python split_sdf.py compounds.sdf --output-dir ligands/

Features:
- Preserves all molecular data and properties
- Uses compound names from SDF properties when available
- Handles duplicates and special characters in filenames
- Shows progress for large files
- Creates clean, individual SDF files ready for docking

Author: SDF splitter for molecular docking workflows
"""

import sys
import re
from pathlib import Path
import argparse

def clean_filename(name):
    """Clean a string to be safe for filename use"""
    # Remove or replace problematic characters
    name = re.sub(r'[<>:"/\\|?*]', '_', name)  # Windows problematic chars
    name = re.sub(r'[\s]+', '_', name)  # Replace spaces with underscores
    name = re.sub(r'[^\w\-_\.]', '', name)   # Keep only alphanumeric, dash, underscore, dot
    name = name.strip('._')  # Remove leading/trailing dots and underscores
    
    # Ensure it's not empty and not too long
    if not name or len(name) < 1:
        name = "compound"
    if len(name) > 50:  # Reasonable filename length limit
        name = name[:50]
    
    return name

def extract_compound_name(sdf_block):
    """Extract compound name from SDF properties"""
    lines = sdf_block.split('\n')
    
    # Try different common property names for compound identification
    name_patterns = [
        r'>\s*<COMPOUND[_\s]*NAME>',
        r'>\s*<NAME>',
        r'>\s*<CHEMBL[_\s]*ID>',
        r'>\s*<ID>',
        r'>\s*<TITLE>',
        r'>\s*<SMILES>',  # Use SMILES as last resort
    ]
    
    for i, line in enumerate(lines):
        for pattern in name_patterns:
            if re.match(pattern, line.strip(), re.IGNORECASE):
                # Get the value from the next non-empty line
                for j in range(i + 1, min(i + 5, len(lines))):
                    if lines[j].strip() and not lines[j].strip().startswith('>'):
                        name = lines[j].strip()
                        if name and len(name) > 1:
                            return clean_filename(name)
                break
    
    # If no name found, try to extract from the first line (molecule name line)
    if len(lines) > 0 and lines[0].strip():
        potential_name = lines[0].strip()
        if len(potential_name) > 0:
            return clean_filename(potential_name)
    
    return None

def split_sdf_file(input_file, output_dir=None):
    """Split multi-compound SDF into individual files"""
    
    input_path = Path(input_file)
    if not input_path.exists():
        print(f"❌ Input file not found: {input_file}")
        return False
    
    # Set up output directory
    if output_dir is None:
        output_dir = input_path.parent / f"{input_path.stem}_split"
    else:
        output_dir = Path(output_dir)
    
    output_dir.mkdir(exist_ok=True)
    print(f"📂 Output directory: {output_dir}")
    
    # Read the entire file
    try:
        with open(input_path, 'r', encoding='utf-8', errors='replace') as f:
            content = f.read()
    except Exception as e:
        print(f"❌ Error reading file: {e}")
        return False
    
    # Split by SDF delimiter ($$$$)
    compounds = content.split('$$$$')
    
    # Remove empty entries
    compounds = [comp.strip() for comp in compounds if comp.strip()]
    
    print(f"🧪 Found {len(compounds)} compounds in SDF file")
    
    if len(compounds) == 0:
        print("❌ No compounds found in file!")
        return False
    
    # Process each compound
    successful = 0
    name_counter = {}  # Track duplicate names
    
    for i, compound_block in enumerate(compounds, 1):
        try:
            # Extract compound name
            compound_name = extract_compound_name(compound_block)
            
            # If no name found, use generic naming
            if not compound_name:
                compound_name = f"compound_{i:04d}"
            
            # Handle duplicates by adding counter
            original_name = compound_name
            if compound_name in name_counter:
                name_counter[compound_name] += 1
                compound_name = f"{original_name}_{name_counter[compound_name]:02d}"
            else:
                name_counter[compound_name] = 0
            
            # Create output filename
            output_file = output_dir / f"{compound_name}.sdf"
            
            # Write individual SDF file
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(compound_block.strip())
                f.write('\n$$$$\n')  # Add SDF terminator
            
            successful += 1
            
            # Progress indicator for large files
            if len(compounds) > 50 and i % 50 == 0:
                print(f"   📝 Processed {i}/{len(compounds)} compounds...")
            elif len(compounds) <= 50:
                print(f"   ✅ {i:3d}. {compound_name}")
        
        except Exception as e:
            print(f"   ❌ Error processing compound {i}: {e}")
            continue
    
    print(f"\n🎉 SUCCESS! Split {successful}/{len(compounds)} compounds")
    print(f"📁 Individual SDF files saved in: {output_dir}")
    
    if successful < len(compounds):
        failed = len(compounds) - successful
        print(f"⚠️  {failed} compounds failed to process")
    
    return True

def main():
    parser = argparse.ArgumentParser(
        description="Split multi-compound SDF file into individual files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python split_sdf.py compounds.sdf
    python split_sdf.py chembl_download.sdf --output-dir my_ligands/
    python split_sdf.py large_dataset.sdf -o compounds/

The script will:
• Create individual SDF files for each compound
• Use compound names from SDF properties when available
• Handle duplicate names automatically
• Preserve all molecular data and properties
• Create filenames safe for all operating systems
        """
    )
    
    parser.add_argument('input_file', help='Input SDF file with multiple compounds')
    parser.add_argument('-o', '--output-dir', help='Output directory (default: input_filename_split/)')
    
    args = parser.parse_args()
    
    print("🧬 SDF FILE SPLITTER")
    print("=" * 30)
    print(f"📄 Input file: {args.input_file}")
    
    if split_sdf_file(args.input_file, args.output_dir):
        print("\n✅ Splitting completed successfully!")
        print("💡 Individual SDF files are ready for molecular docking!")
        return 0
    else:
        print("\n❌ Splitting failed!")
        return 1

if __name__ == "__main__":
    sys.exit(main())
