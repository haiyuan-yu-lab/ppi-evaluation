import numpy as np
import os
from util import *  
from Bio.PDB.MMCIFParser import MMCIFParser
from Bio import SeqIO
import json

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

def extract_sequences_from_json(job_request_file):
    """
    Extract protein sequences from the job request JSON file.

    Args:
        job_request_file (str): Path to the job request JSON file.

    Returns:
        tuple: Sequence for prot1 and prot2.
    """
    sequences = []
    if os.path.exists(job_request_file):
        try:
            with open(job_request_file, 'r') as f:
                data = json.load(f)

            # Extract sequences from JSON structure
            for entry in data:
                for seq_entry in entry.get("sequences", []):
                    seq = seq_entry.get("proteinChain", {}).get("sequence", "")
                    if seq:
                        sequences.append(seq)

        except Exception as e:
            log_message(f"Error reading JSON file {job_request_file}: {e}")
    else:
        log_message(f"Job request JSON file not found: {job_request_file}")

    # Ensure we have two sequences
    if len(sequences) < 2:
        log_message(f"Less than two sequences found in {job_request_file}")

    return sequences[0] if len(sequences) > 0 else "", sequences[1] if len(sequences) > 1 else ""
