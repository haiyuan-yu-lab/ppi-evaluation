import os
import zipfile
from Bio import SeqIO
import json     
import sys

# Add the project root (ppi-evaluation/) to sys.path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from util import *
from compute_features import *
from feature_extraction import *                

def process_zip(zip_folder, output_dir, metrics_df):
    
    zip_files = [f for f in os.listdir(zip_folder) if f.endswith('.zip')]
 
    for zip_file in zip_files:
        zip_path = os.path.join(zip_folder, zip_file)
        zip_name = zip_file.replace('.zip', '')
        protein_a, protein_b = zip_name.split('_')[1:3]
        
        # Extract sequences for prot1 and prot2
        #seq1, seq2 = read_fasta_sequence(protein_a, protein_b)

        nonstr_label, str_label = categorize_predictions(protein_a, protein_b)
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(output_dir)

        # Paths to extracted files
        cif_file = os.path.join(output_dir, f"{zip_name}_model_0.cif")
        full_data_json = os.path.join(output_dir, f"{zip_name}_full_data_0.json")
        summary_json = os.path.join(output_dir, f"{zip_name}_summary_confidences_0.json")
        job_request_json = os.path.join(output_dir, f"{zip_name}_job_request.json")

        if not (os.path.exists(cif_file) and os.path.exists(summary_json) and os.path.exists(full_data_json)):
            print(f"Files missing for {zip_name}")
            continue

        # Extract sequences for prot1 and prot2
        seq1, seq2 = extract_sequences_from_json(job_request_json)
        print(f"seq1: {seq1}, seq2: {seq2}")

        # Extract pLDDT and PAE
        with open(full_data_json, 'r') as f:
            full_data = json.load(f)

        plddt = np.mean(np.array(full_data['atom_plddts']))
        pae_matrix = np.array(full_data['pae'])
        mean_pae = np.mean(pae_matrix)


        # Calculate LIS
        chain_a_len = len(pae_matrix) // 2
        lis_score = compute_lis(pae_matrix, chain_a_len)

        # Read CIF file and calculate pDockQ
        chain_coords, chain_plddt = read_cif(cif_file)
        pDockQ, _ = calc_pdockq(chain_coords, chain_plddt, 8)
        pDockQ2 = compute_pdockq2(plddt, mean_pae)

        # Load summary for additional metrics
        with open(summary_json, 'r') as f:
            summary_data = json.load(f)
        iptm = summary_data.get('iptm', None)
        ptm = summary_data.get('ptm', None)
        ranking_confidence = summary_data.get('ranking_score', None)

        # Add data to DataFrame if all required metrics are available
        if None not in (pDockQ, pDockQ2, lis_score, iptm, ptm, ranking_confidence):
            metrics_df = metrics_df.append({
                'prot1': protein_a,
                'prot2': protein_b,
                'sequence_A': seq1,
                'sequence_B': seq2,
                'nonstr_label': nonstr_label,
                'str_label': str_label,
                'pLDDT': round(plddt, 3),
                'pAE': round(mean_pae, 3),
                'pDockQ': pDockQ,
                'pDockQ2': pDockQ2,
                'LIS': lis_score,
                'ipTM': round(float(iptm), 3),
                'pTM': round(float(ptm), 3),
                'ranking_score': round(float(ranking_confidence), 2)
            }, ignore_index=True)


    return metrics_df


def create_af3_dataset(zip_folder, output_dir):
    """
    Processes all AF3 .zip files to generate a complete DataFrame with all features and sequences.
    """
    metrics_df = pd.DataFrame(columns=['prot1', 'prot2', 'sequence_A', 'sequence_B',
                                       'nonstr_label', 'str_label', 'pLDDT', 'pAE',
                                       'pDockQ', 'pDockQ2', 'LIS', 'ipTM', 'pTM',
                                       'ranking_score'])
    
    metrics_df = process_zip(zip_folder, output_dir, metrics_df) #(zip_folder, zip_file, output_dir, metrics_df)

    return metrics_df
