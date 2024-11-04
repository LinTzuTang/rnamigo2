import os
import gemmi
import pandas as pd
import argparse

def convert_pdb_to_cif(pdb_file, output_dir):
    # Load the PDB file using gemmi
    try:
        structure = gemmi.read_structure(pdb_file)
        base_filename = os.path.basename(pdb_file).replace('.pdb', '')
        
        for model in structure:
            for chain in model:
                chain.name = 'A'
        
        # Save as mmCIF format
        output_file = os.path.join(output_dir, f'{base_filename}.cif')
        
        # Write the structure to CIF format
        structure.make_mmcif_document().write_file(output_file)
        
        print(f"Converted {pdb_file} to CIF format at {output_file}")
    except Exception as e:
        print(f"Failed to convert {pdb_file} to CIF: {e}")


def batch_convert(pdb_dir, output_dir):
    pdb_files = [f for f in os.listdir(pdb_dir) if f.endswith('.pdb')]
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for pdb_file in pdb_files:
        pdb_path = os.path.join(pdb_dir, pdb_file)
        convert_pdb_to_cif(pdb_path, output_dir)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert PDB files to CIF format.')
    parser.add_argument('--pdb_dir', type=str, required=True, help='Directory containing PDB files')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save CIF files')
    
    args = parser.parse_args()
    
    batch_convert(args.pdb_dir, args.output_dir)
