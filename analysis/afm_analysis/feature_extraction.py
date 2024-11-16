import numpy as np
import os
from util import *  
from Bio import SeqIO
import json
from Bio.PDB import PDBParser
import pandas as pd



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


def get_lowest_ranked_pdb(folder_path):
    """
    Find the lowest-ranked PDB file in the given folder.
    If none are found, prioritize 'relaxed_model_{number}_multimer_...pdb'.
    If none are found, fall back to 'unrelaxed_model_...pdb'.
    """
    # Look for ranked_*.pdb files
    ranked_files = [
        f for f in os.listdir(folder_path) if f.startswith("ranked_") and f.endswith(".pdb")
    ]
    if ranked_files:
        ranked_files.sort(key=lambda x: int(x.split('_')[1].split('.')[0]))
        return os.path.join(folder_path, ranked_files[0])

    # Look for relaxed_model_{number}_multimer_...pdb files
    relaxed_files = [
        f for f in os.listdir(folder_path) if f.startswith("relaxed_model_") and f.endswith(".pdb")
    ]
    if relaxed_files:
        relaxed_files.sort(key=lambda x: int(x.split('_')[2]))
        return os.path.join(folder_path, relaxed_files[0])

    # Look for unrelaxed_model_...pdb files
    unrelaxed_files = [
        f for f in os.listdir(folder_path) if f.startswith("unrelaxed_model_") and f.endswith(".pdb")
    ]
    if unrelaxed_files:
        unrelaxed_files.sort(key=lambda x: int(x.split('_')[2]))
        return os.path.join(folder_path, unrelaxed_files[0])

    # If no PDB files are found
    return None



# Function to assign structural and nonstructural labels for catergorizations.
def categorize_predictions(pred_protein_a, pred_protein_b):
    pred_protein_a = pred_protein_a.upper()
    pred_protein_b = pred_protein_b.upper()
    ppi = pred_protein_a + ':' + pred_protein_b
    print(ppi)

    label_data = pd.read_csv('/home/yl986/data/protein_interaction/results/af3_ppi_with_label.csv')

    for _, row in label_data.iterrows():
        ppi_ref = row['ppi']
        if ppi_ref == ppi:
            nonstr_label = row['non_struct_label']
            str_label = row['str_label']
            return nonstr_label, str_label



############################################ Computing AF prediction metrics (pDockQ, LIS, pDockQ2) ##########################


# Function to compute LIS score based on the reference code
def compute_lis(pae_matrix, chain_a_len, pae_cutoff=12.0):
    try:
        thresholded_pae = np.where(pae_matrix < pae_cutoff, 1, 0)
        scaled_pae = reverse_and_scale_matrix(pae_matrix, pae_cutoff)

        selected_values_interaction1_score = scaled_pae[:chain_a_len, chain_a_len:][thresholded_pae[:chain_a_len, chain_a_len:] == 1]
        average_selected_interaction1_score = np.mean(selected_values_interaction1_score) if selected_values_interaction1_score.size > 0 else 0

        selected_values_interaction2_score = scaled_pae[chain_a_len:, :chain_a_len][thresholded_pae[chain_a_len:, :chain_a_len] == 1]
        average_selected_interaction2_score = np.mean(selected_values_interaction2_score) if selected_values_interaction2_score.size > 0 else 0

        average_selected_interaction_total_score = (average_selected_interaction1_score + average_selected_interaction2_score) / 2

        lis = round(average_selected_interaction_total_score, 3)

        return lis
    except Exception as e:
        return None


# Function to reverse and scale matrix
def reverse_and_scale_matrix(matrix: np.ndarray, pae_cutoff: float = 12.0) -> np.ndarray:
    scaled_matrix = (pae_cutoff - matrix) / pae_cutoff
    return np.clip(scaled_matrix, 0, 1)

# Function to calculate pDockQ
def calc_pdockq(chain_coords, chain_plddt, t):

    # Get coords and plddt per chain
    ch1, ch2 = [*chain_coords.keys()]
    coords1, coords2 = chain_coords[ch1], chain_coords[ch2]
    plddt1, plddt2 = chain_plddt[ch1], chain_plddt[ch2]

    # Calculate 2-norm
    mat = np.append(coords1, coords2, axis=0)
    a_min_b = mat[:, np.newaxis,:] - mat[np.newaxis,:,:]
    dists = np.sqrt(np.sum(a_min_b.T ** 2, axis =0)).T
    l1 = len(coords1)
    contact_dists = dists[:l1,l1:] # Upper triangular -> first dim = chain 1
    contacts = np.argwhere(contact_dists<=t)

    if contacts.shape[0] < 1:
        pdockq = 0
        ppv = 0
    else:
        #Get the average interface plDDT
        avg_if_plddt = np.average(np.concatenate([plddt1[np.unique(contacts[:,0])], plddt2[np.unique(contacts[:,1])]]))

        #Get the number of interface contacts
        n_if_contacts = contacts.shape[0]
        x = avg_if_plddt*np.log10(n_if_contacts)
        pdockq = 0.724 / (1 + np.exp(-0.052*(x-152.611)))+0.018

        #PPV
        PPV = np.array([0.98128027, 0.96322524, 0.95333044, 0.9400192 ,
            0.93172991, 0.92420274, 0.91629946, 0.90952562, 0.90043139,
            0.8919553 , 0.88570037, 0.87822061, 0.87116417, 0.86040801,
            0.85453785, 0.84294946, 0.83367787, 0.82238224, 0.81190228,
            0.80223507, 0.78549007, 0.77766077, 0.75941223, 0.74006263,
            0.73044282, 0.71391784, 0.70615739, 0.68635536, 0.66728511,
            0.63555449, 0.55890174])

        pdockq_thresholds = np.array([0.67333079, 0.65666073, 0.63254566, 0.62604391,
            0.60150931, 0.58313803, 0.5647381 , 0.54122438, 0.52314392,
            0.49659878, 0.4774676 , 0.44661346, 0.42628389, 0.39990988,
            0.38479715, 0.3649393 , 0.34526004, 0.3262589 , 0.31475668,
            0.29750023, 0.26673725, 0.24561247, 0.21882689, 0.19651314,
            0.17606258, 0.15398168, 0.13927677, 0.12024131, 0.09996019,
            0.06968505, 0.02946438])
        inds = np.argwhere(pdockq_thresholds>=pdockq)

        if len(inds)>0:
            ppv = PPV[inds[-1]][0]
        else:
            ppv = PPV[0]

    return pdockq, ppv

# Function to compute pDockQ2
def compute_pdockq2(plddt, pae):
    L = 1.31
    x0 = 84.733
    k = 0.075
    b = 0.005
    pDockQ2 = L / (1 + np.exp(-k * (plddt - pae / 10 - x0))) + b
    return pDockQ2




