import sys
import os
import numpy as np
from Bio.PDB.MMCIFParser import MMCIFParser
from Bio.PDB import PDBParser
from Bio import SeqIO
import pandas as pd
from pathlib import Path


PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))


# Analysis path
AF3_ANALYSIS_PATH = os.path.join(PROJECT_ROOT, "analysis/af3_analysis")
AFM_ANALYSIS_PATH = os.path.join(PROJECT_ROOT, "analysis/afm_analysis")


# Data path
AF_SCORES = os.path.join(PROJECT_ROOT, "data")

# Missing log_file_path
MISSING_PPI_LABEL = os.path.join(AF_SCORES, "missing_ppi_label.txt")

# Results Paths
AF3_RESULTS_CSV = os.path.join(AF_SCORES, "af3_results.csv")
AFM_RESULTS_CSV = os.path.join(AF_SCORES, "afm_results.csv")

# Input path
FASTA_PATH = "/home/yl986/data/protein_interaction/input_fasta"

# AF3 prediction folder
AF3_PRED_FOLDER = "/share/yu/ppi_pred/af3/af3-results"
AFM_PRED_FOLDER = "/home/yl986/data/afm_results"

AF3_OUTPUT_DIR = os.path.join(AF3_PRED_FOLDER, "output")
AFM_OUTPUT_DIR = os.path.join(AFM_ANALYSIS_PATH, "output")



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


def normalize_scores(df):
    """
    We aim to normalize prediction scores that are out of the [0.0, 1.0] range.
    """

    norm_scores = ['mean_pLDDT', 'mean_pAE','mean_interface_pLDDT', 'mean_interface_pAE', 'ranking_score']

    df['mean_pAE'] = 1 - (df['mean_pAE'] / 36.0)
    df['mean_interface_pAE'] = 1 - (df['mean_interface_pAE'] / 36.0)
    df['mean_pLDDT'] = df['mean_pLDDT'] / 100.0
    df['mean_interface_pLDDT'] = df['mean_interface_pLDDT'] / 100.0 
    df['ranking_score'] = (df['ranking_score'] + 100.0) / (1.5 + 100.0)

    return df





def build_af3_ppi_with_label(output_path):
    """
    Build the af3_ppi_with_label.csv file efficiently, minimizing memory usage.

    Parameters:
    - output_path (str): Path where the resulting CSV file will be saved.

    Returns:
    - None: Saves the output CSV file to the specified path.
    """
    # Paths to input files
    data_root = Path('/home/yl986/data')
    parsed_root = data_root / 'protein_interaction/parsed'
    hint_root = data_root / 'HINT/update_2024/outputs'
    ires_root = data_root / 'IRES/parsed_files/20241018'

    # Load the interaction sets from BioGRID, IntAct, and STRING (Exclusion set)
    ppi_comb3 = pd.read_csv(parsed_root / 'cache/ppi_comb3_rev.tsv', sep='\t', dtype={'evidence_code': str, 'pubmed': str}, usecols=['ppi'])

    # Updated with appended HINT (2024.6.27)
    ppi_comb4 = pd.read_csv(parsed_root / 'cache/ppi_comb4.csv', dtype={'evidence_code': str, 'gene_name1': str, 'gene_name2': str, 'pubmed': str}, usecols=['ppi']) 

    # Positive sets
    hint_lcb24 = pd.read_csv(hint_root / 'HINT_format/taxa/HomoSapiens/HomoSapiens_lcb_hq.txt', sep='\t', dtype=str)

    # Positive sets
    hint_lcb24['ppi'] = hint_lcb24.apply(lambda x: ':'.join(sorted([x['Uniprot_A'], x['Uniprot_B']])), axis=1)
    hint_lcb24['ppi'] = hint_lcb24.apply(lambda x: ':'.join(sorted([x['Uniprot_A'], x['Uniprot_B']])), axis=1)

    # Prepare nonstructural lists
    nonstr_exclusion = pd.concat([ppi_comb3[['ppi']], ppi_comb4[['ppi']]])['ppi'].drop_duplicates().tolist()
    nonstr_pos = hint_lcb24['ppi'].drop_duplicates().tolist()

    
    # Complex-based (structural)
    ires_human = pd.read_csv(ires_root / 'ires_human_all.csv', usecols=['ppi'])
    ppi_by_pdb = pd.read_csv(ires_root / 'ppi_by_pdb_info.csv', usecols=['ppi'])

    ppi_in_pdb = ppi_by_pdb['ppi'].drop_duplicates().tolist()
    ppi_str = ires_human['ppi'].drop_duplicates().tolist()


    all_ppis = set(nonstr_exclusion + nonstr_pos + ppi_in_pdb + ppi_str)
    df_ppi = pd.DataFrame({'ppi': list(all_ppis)})

    # Assign labels using the helper function
    df_labeled_ppi = assign_ppi_label(df_ppi, nonstr_pos, nonstr_exclusion, ppi_str, ppi_in_pdb)

    # Save the labeled DataFrame to the specified output path
    df_labeled_ppi.to_csv(output_path, index=False)



def assign_ppi_label(df_ppi, nonstr_pos, nonstr_ex, str_pos, ppi_in_pdb):
    df_ppi = df_ppi.copy()
    # non-structural label
    df_ppi['non_struct_label'] = -1
    df_ppi.loc[~df_ppi['ppi'].isin(nonstr_ex), 'non_struct_label'] = 0
    df_ppi.loc[df_ppi['ppi'].isin(nonstr_pos), 'non_struct_label'] = 1

    # structural label
    df_ppi['str_label'] = -1  # interactions not found in the same complex structure / no structural evidence
    df_ppi.loc[df_ppi['ppi'].isin(ppi_in_pdb), 'str_label'] = 0  # found in same complex
    df_ppi.loc[df_ppi['ppi'].isin(str_pos), 'str_label'] = 1  # structural evidence
    
    label_dict = {-1: 'unclear (exclude for analysis)', 0: 'non-interacting pairs', 1: 'HINT-binary-HQ-LC'}
    df_ppi['nonstr_label_name'] = df_ppi['non_struct_label'].apply(lambda x: label_dict.get(x, x))

    label_dict = {-1: 'not found in same PDB complex', 0: 'no physical interaction', 1: 'direct physical interaction'}
    df_ppi['str_label_name'] = df_ppi['str_label'].apply(lambda x: label_dict[x])
    
    return df_ppi



# Debugging purpose
def log_message(message):
    print(f"[LOG]: {message}")



