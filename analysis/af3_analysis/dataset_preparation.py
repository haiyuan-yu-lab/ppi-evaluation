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

def process_zip(zip_folder, output_dir, metrics_df, log_file_path):
    
    zip_files = [f for f in os.listdir(zip_folder) if f.endswith('.zip')]


    for zip_file in zip_files:
        zip_path = os.path.join(zip_folder, zip_file)
        zip_name = zip_file.replace('.zip', '')
        protein_a, protein_b = zip_name.split('_')[1:3]
        
        # Exclude homo-dimers from our analysis
        if protein_a == protein_b:
            print(f"Homo-dimer {protein_a}_{protein_b} excluded!")
            continue

        # Extract labels 
        nonstr_label, str_label = categorize_predictions(protein_a, protein_b)
        
        # If both labels are missing, log the entry to the text file
        if nonstr_label is None and str_label is None:
            with open(log_file_path, 'a') as log_file:
                log_file.write(f"{protein_a}_{protein_b}\n")
            print(f"Logged missing labels for {protein_a}_{protein_b}")
            continue

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
        
        # Get the structure of a MMCIF file
        structure = get_structure(cif_file)
        chains = [chain.id for chain in structure[0]]
    
        # Process all chains iteratively
        plddt_lst = []
        remain_contact_lst = []


        for idx in range(len(chains)):
            main_chain = chains[idx]
            contact_chains = list(set(chains) - {main_chain})

            IF_plddt, contact_lst = retrieve_IFplddt(
                structure, main_chain, contact_chains, 8
            )


            plddt_lst.append(IF_plddt)
            remain_contact_lst.append(contact_lst)


        # Aggregate mean interface metrics
        mean_IF_pLDDT = np.mean(plddt_lst)
        
        avgif_pae = retrieve_IFPAEinter(structure, pae_matrix, remain_contact_lst, 8)

        pdockq2 = calc_pmidockq(avgif_pae, plddt_lst)['pmidockq'].mean()
        
        ## Print output 
        print(f"pDockQ2: {pdockq2}")
        print(f"interface pLDDT: {plddt_lst}")
        print(f"mean interface PAE: {avgif_pae}")

        # Calculate LIS
        chain_a_len = len(pae_matrix) // 2
        lis_score = compute_lis(pae_matrix, chain_a_len)

        # Read CIF file and calculate pDockQ
        chain_coords, chain_plddt = read_cif(cif_file)
        pDockQ, _ = calc_pdockq(chain_coords, chain_plddt, 8)

        # Load summary for additional metrics
        with open(summary_json, 'r') as f:
            summary_data = json.load(f)
        iptm = summary_data.get('iptm', None)
        ptm = summary_data.get('ptm', None)
        ranking_confidence = summary_data.get('ranking_score', None)

        # Add data to DataFrame if all required metrics are available
        if None not in (pDockQ, pdockq2, lis_score, iptm, ptm, ranking_confidence):
            row = pd.DataFrame([{
                'prot1': protein_a,
                'prot2': protein_b,
                'sequence_A': seq1,
                'sequence_B': seq2,
                'nonstr_label': nonstr_label,
                'str_label': str_label,
                'mean_pLDDT': round(plddt, 3),
                'mean_pAE': round(mean_pae, 3),
                'mean_interface_pLDDT': round(mean_IF_pLDDT, 3),
                'mean_interface_pAE': round(np.mean(avgif_pae), 3),
                'pDockQ': pDockQ,
                'pDockQ2': round(pdockq2, 3),
                'LIS': lis_score,
                'ipTM': round(float(iptm), 3),
                'pTM': round(float(ptm), 3),
                'ranking_score': round(float(ranking_confidence), 2)
            }])

            metrics_df = pd.concat([metrics_df, row],ignore_index=True)

    return metrics_df


def create_af3_dataset(zip_folder, output_dir):
    """
    Processes all AF3 .zip files to generate a complete DataFrame with all features and sequences.
    """
    metrics_df = pd.DataFrame(columns=['prot1', 'prot2', 'sequence_A', 'sequence_B',
                                       'nonstr_label', 'str_label', 'mean_pLDDT', 'mean_pAE',
                                       'mean_interface_pLDDT', 'mean_interface_pAE',
                                       'pDockQ', 'pDockQ2', 'LIS', 'ipTM', 'pTM',
                                       'ranking_score'])


    metrics_df = process_zip(zip_folder, output_dir, metrics_df, MISSING_PPI_LABEL)

    return metrics_df
