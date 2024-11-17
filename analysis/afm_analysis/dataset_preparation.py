import os
import zipfile
import pandas as pd
from Bio import SeqIO
import json     
import pickle
import sys
import numpy as np

# Add the project root (ppi-evaluation/) to sys.path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from util import read_pdb, log_message
from compute_features import compute_lis, calc_pdockq, compute_pdockq2, categorize_predictions
from feature_extraction import get_lowest_ranked_pdb

def process_afm_folder(folder_path, metrics_df):
    """
    Process a single AlphaFold-Multimer prediction folder to extract relevant metrics.
    """

    folder_name = os.path.basename(folder_path)
    
    prot1, prot2 = folder_name.split('_') # example - prot1: P06870; prot2: Q7RTU4

    nonstr_label, str_label = categorize_predictions(prot1, prot2) 
    
    print(f"nonstr_label: {nonstr_label}")
    print(f"str_label: {str_label}")

    pdb_file = get_lowest_ranked_pdb(folder_path)
    result_pkl_file = next(
        (os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.startswith("result_") and f.endswith(".pkl")),
        None,
    )

    print(f"pdb_file: {pdb_file}")

    chain_coords, chain_plddt, chain_sequences = read_pdb(pdb_file)
    
    print(f"[DEBUG] chain_coords: {chain_coords.keys()}")
    print(f"[DEBUG] chain_plddt: {chain_plddt.keys()}")
    print(f"[DEBUG] chain_sequences: {chain_sequences}")
    
   
    chain_keys = list(chain_sequences.keys())
    
    seq1 = ""
    seq2 = ""

    if len(chain_keys) >= 2:
        prot1_chain, prot2_chain = chain_keys[0], chain_keys[1]
        seq1, seq2 = chain_sequences[prot1_chain], chain_sequences[prot2_chain]
    
    print(f"seq1: {seq1}")
    print(f"seq2: {seq2}")

    # Check if all required files exist
    if not all(os.path.exists(f) for f in [pdb_file, result_pkl_file]):
        log_message(f"Files missing in {folder_path}. Skipping.")
        return metrics_df

    # Extract pLDDT and PAE (if available from result_pkl_file)
    
    try:
        with open(result_pkl_file, 'rb') as f:
            result_data = pickle.load(f)
        plddt = np.mean(np.array(result_data['plddt']))
        pae_matrix = np.array(result_data.get('predicted_aligned_error', []))
        mean_pae = np.mean(pae_matrix) if len(pae_matrix) > 0 else None
    
        # Check for `ranking_confidence`
        ranking_score = result_data.get('ranking_confidence', None)
        if ranking_score is None:
            log_message(f"ranking_confidence missing in {result_pkl_file}. Defaulting to None.")

    except Exception as e:
        log_message(f"Error reading result file {result_pkl_file}: {e}")
        return metrics_df

    # Extract LIS
    chain_a_len = len(pae_matrix) // 2
    lis_score = compute_lis(pae_matrix, chain_a_len)
    
    print(f"LIS: {lis_score}")
    # Calculate pDockQ and pDockQ2
    pDockQ, _ = calc_pdockq(chain_coords, chain_plddt, 8)
    pDockQ2 = compute_pdockq2(plddt, mean_pae)

    print(f"pDockQ: {pDockQ}")
    print(f"pDockQ2: {pDockQ2}")

    # Add to DataFrame if all metrics are available
    metrics_df = pd.concat([
        metrics_df,
        pd.DataFrame([{
            'prot1': prot1,
            'prot2': prot2,
            'sequence_A': seq1,
            'sequence_B': seq2,
            'nonstr_label': nonstr_label, 
            'str_label': str_label,
            'pLDDT': round(plddt, 3),
            'pAE': round(mean_pae, 3) if mean_pae is not None else None,
            'pDockQ': pDockQ,
            'pDockQ2': pDockQ2,
            'LIS': lis_score,
            'ranking_score': round(float(ranking_score), 2) if ranking_score else None
        }])
    ], ignore_index=True)

    return metrics_df


def create_afm_dataset(folder_path):
    """
    Process all AlphaFold-Multimer prediction folders and generate a DataFrame with features.
    """

    metrics_df = pd.DataFrame(columns=['prot1', 'prot2', 'sequence_A', 'sequence_B',
                                       'nonstr_label', 'str_label', 'pLDDT', 'pAE',
                                       'pDockQ', 'pDockQ2', 'LIS', 'ipTM', 'pTM',
                                       'ranking_score'])

    afm_folders = [
        os.path.join(folder_path, d) for d in os.listdir(folder_path)
        if os.path.isdir(os.path.join(folder_path, d))
    ]

    for folder in afm_folders:
        metrics_df = process_afm_folder(folder, metrics_df)

    return metrics_df
