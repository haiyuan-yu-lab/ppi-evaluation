import sys
import os
import numpy as np
from Bio.PDB.MMCIFParser import MMCIFParser
from Bio.PDB import PDBParser
from Bio import SeqIO



PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))


# Analysis path
AF3_ANALYSIS_PATH = os.path.join(PROJECT_ROOT, "analysis/af3_analysis")
AFM_ANALYSIS_PATH = os.path.join(PROJECT_ROOT, "analysis/afm_analysis")

# Output Paths
AF3_OUTPUT_DIR = os.path.join(AF3_ANALYSIS_PATH, "output")
AFM_OUTPUT_DIR = os.path.join(AFM_ANALYSIS_PATH, "output")


# Data path
AF_SCORES = os.path.join(PROJECT_ROOT, "data")

# Missing log_file_path
MISSING_PPI_LABEL = os.path.join(AF_scores, "missing_ppi_label.txt")

# Results Paths
AF3_RESULTS_CSV = os.path.join(AF_SCORES, "af3_results.csv")
AFM_RESULTS_CSV = os.path.join(AF_SCORES, "afm_results.csv")

# Input path
FASTA_PATH = "/home/yl986/data/protein_interaction/input_fasta"

# AF3 prediction folder
AF3_PRED_FOLDER = "/share/yu/ppi_pred/af3/af3-results"
AFM_PRED_FOLDER = "/home/yl986/data/afm_results"

# Visualization Colors
COLORS = {
    "nonstruct_neg": "#EF4035",
    "nonstruct_pos": "#6EB43F",
    "struct_neg": "#F8981D",
    "struct_pos": "#006699"
}

# Feature Columns
FEATURE_COLUMNS = ['LIS', 'ipTM', 'pTM', 'pLDDT', 'pDockQ', 'pAE', 'pDockQ2', 'ranking_score']



########################################## Any utility or helper functions ##################################################

# Function to parse CIF file and extract chain coordinates and pLDDT values (for all atoms)
def read_cif(cif_file):
    '''
    Read a .cif file to contain all chains.
    '''
    chain_coords, chain_plddt = {}, {}
    try:
        parser = MMCIFParser(QUIET=True)
        structure = parser.get_structure("structure", cif_file)

        for model in structure:
            for chain in model:
                for residue in chain:
                    for atom in residue:
                        if atom.get_name() == 'CB' or (atom.get_name() == 'CA' and residue.get_resname() == 'GLY'):
                            if chain.id in chain_coords:
                                chain_coords[chain.id].append(atom.get_coord())
                                chain_plddt[chain.id].append(atom.get_bfactor())  # Assuming pLDDT values are stored in the B-factor field
                            else:
                                chain_coords[chain.id] = [atom.get_coord()]
                                chain_plddt[chain.id] = [atom.get_bfactor()]

        # Convert to arrays
        for chain in chain_coords:
            chain_coords[chain] = np.array(chain_coords[chain])
            chain_plddt[chain] = np.array(chain_plddt[chain])

        return chain_coords, chain_plddt

    except Exception as e:
        print(f"Error parsing CIF file {cif_file}: {e}")
        return None, None


# Function to parse MMCIF file and return a structure object
def get_structure(cif_file):
    
    parser = MMCIFParser(QUIET=True)

    cif_id = cif_file.split('/')[-1].split('.')[0]  

    try:
        structure = parser.get_structure(cif_id, cif_file)
        return structure
    except Exception as e: 
        print(f"Error parsing {cif_file}")
        return None


def read_pdb(pdb_file):
    """
    Parse a PDB file to extract chain coordinates, plDDT scores, and sequences.
    """
    chain_coords = {}
    chain_plddt = {}
    chain_sequences = {}

    try:
        with open(pdb_file, 'r') as f:
            current_chain = None

            for line in f:
                if line.startswith("ATOM"):
                    # Extract chain, coordinates, and plDDT
                    chain = line[21].strip()
                    x, y, z = map(float, (line[30:38], line[38:46], line[46:54]))
                    plddt = float(line[60:66].strip())
                    res_id = line[17:20].strip()  # Residue 3-letter code

                    # Initialize the chain if not already present
                    if chain not in chain_coords:
                        chain_coords[chain] = []
                        chain_plddt[chain] = []
                        chain_sequences[chain] = []

                    # Append coordinates, plDDT, and sequence
                    chain_coords[chain].append([x, y, z])
                    chain_plddt[chain].append(plddt)
                    one_letter_residue = residue_to_one_letter(res_id)
                    if one_letter_residue:
                        chain_sequences[chain].append(one_letter_residue)

            # Convert lists to arrays and sequences to strings
            for chain in chain_coords:
                chain_coords[chain] = np.array(chain_coords[chain])
                chain_plddt[chain] = np.array(chain_plddt[chain])
                chain_sequences[chain] = "".join(chain_sequences[chain])

    except Exception as e:
        log_message(f"Error reading PDB file {pdb_file}: {e}")

    return chain_coords, chain_plddt, chain_sequences


def residue_to_one_letter(residue):
    """
    Helper method to convert three-letter residue codes to one-letter codes.
    """
    three_to_one_map = {
        "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
        "GLU": "E", "GLN": "Q", "GLY": "G", "HIS": "H", "ILE": "I",
        "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
        "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
    }
    return three_to_one_map.get(residue, None)



def delete_rows(df):
    """
    Removes rows from the DataFrame where the values in 'prot1' and 'prot2' columns are identical.

    Args:
        df (pd.DataFrame): The input DataFrame.

    Returns:
        pd.DataFrame: A DataFrame with the specified rows removed.
    """
    # Keep rows where 'prot1' is not equal to 'prot2'
    filtered_df = df[df['prot1'] != df['prot2']].reset_index(drop=True)
    return filtered_df



# Debugging purpose
def log_message(message):
    print(f"[LOG]: {message}")



