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
    
    print(f"Number of AF3 prediction folders: {len(zip_files)}")

    for zip_file in zip_files:
        zip_path = os.path.join(zip_folder, zip_file)
        zip_name = zip_file.replace('.zip', '')
        protein_a, protein_b = zip_name.split('_')[1:3]
        
        # Exclude homo-dimers from our analysis
        if protein_a == protein_b:
            #print(f"Homo-dimer {protein_a}_{protein_b} excluded!")
            continue

        # Extract labels 
        nonstr_label, str_label = categorize_predictions(protein_a, protein_b)

        # Record missing PPI labels
        logged_entries = set()

        # If both labels are missing, log the entry to the text file
        if nonstr_label is None and str_label is None:
            entry = f"{protein_a}_{protein_b}"
            if entry not in logged_entries:
                with open(log_file_path, 'a') as log_file:
                    log_file.write(entry + "\n")
                logged_entries.add(entry)
            continue

        # Extract AF3 prediction .zip files
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(output_dir)

        # Paths to extracted files
        cif_file = os.path.join(output_dir, f"{zip_name}_model_0.cif")
        full_data_json = os.path.join(output_dir, f"{zip_name}_full_data_0.json")
        summary_json = os.path.join(output_dir, f"{zip_name}_summary_confidences_0.json")
        job_request_json = os.path.join(output_dir, f"{zip_name}_job_request.json")
        
        json_data = json.load(open(full_data_json, 'rb'))

        if not (os.path.exists(cif_file) and os.path.exists(summary_json) and os.path.exists(full_data_json)):
            #print(f"Files missing for {zip_name}")
            continue
        
        token_chain_ids = json_data['token_chain_ids']
        chain_residue_counts = Counter(token_chain_ids)
        subunit_number = list(chain_residue_counts.values())
        subunit_sizes = subunit_number

    

        # Extract sequences for prot1 and prot2
        seq1, seq2 = extract_sequences_from_json(job_request_json)
        #print(f"seq1: {seq1}, seq2: {seq2}")

        # Extract mean pLDDT and PAE scores 
        with open(full_data_json, 'r') as f:
            full_data = json.load(f)

        mean_plddt = np.mean(np.array(full_data['atom_plddts']))
        best_plddt = np.max(np.array(full_data['atom_plddts']))

        pae_matrix = np.array(full_data['pae'])
        mean_pae = np.mean(pae_matrix)
       
        print(f"Dimension of PAE matrix: {pae_matrix.shape}")

        # Transform the PAE matrix
        transformed_pae_matrix = transform_pae_matrix(pae_matrix)
        transformed_pae_matrix = np.nan_to_num(transformed_pae_matrix)
        lia_map = np.where(transformed_pae_matrix > 0, 1, 0)

        # Calculate mean LIS matrix
        mean_lis_matrix = calculate_mean_lis(transformed_pae_matrix, subunit_sizes)
        mean_lis_matrix = np.nan_to_num(mean_lis_matrix)

        print(mean_lis_matrix) 
       
        # From our nxn mean_lis_matrix, extract the mean & best LIS scores.
        mean_lis_score = extract_lis_score(mean_lis_matrix)
        best_lis_score = np.max(mean_lis_matrix)

        # Calculate contact map
        contact_map = calculate_contact_map(cif_file)
        combined_map = np.where((transformed_pae_matrix > 0) & (contact_map == 1), transformed_pae_matrix, 0)
 
        # Calculate the mean LIS matrix
        mean_clis_matrix = calculate_mean_lis(combined_map, subunit_sizes)
        mean_clis_matrix = np.nan_to_num(mean_clis_matrix)

        lia_matrix, lir_matrix, clia_matrix, clir_matrix = get_lia_lir_matrices(subunit_sizes, lia_map, combined_map)
        print(f"lia_matrix: {lia_matrix}")
        mean_lia_score = extract_lis_score(lia_matrix)
        best_lia_score = np.max(mean_lia_score)

        # Get the structure of a MMCIF file
        structure = get_structure(cif_file)
        chains = [chain.id for chain in structure[0]]
    
        # Process all chains iteratively
        plddt_lst = []
        remain_contact_lst = []

        # Extract pLDDT & PAE scores of the PPI-interface.
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
        print(f"avgif_pae: {avgif_pae}")
        pmidockq = calc_pmidockq(avgif_pae, plddt_lst)['pmidockq']
        
        # Average pDockQ2
        mean_pdockq2 = np.mean(pmidockq)
        best_pdockq2 = np.max(pmidockq)

        ## Print output 
        print(f"pDockQ2: {mean_pdockq2}")
        print(f"interface pLDDT: {plddt_lst}")
        print(f"mean interface PAE: {avgif_pae}")


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
        if None not in (pDockQ, mean_pdockq2, mean_lis_score, iptm, ptm, ranking_confidence):
            row = pd.DataFrame([{
                'prot_A': protein_a,
                'prot_B': protein_b,
                'sequence_A': seq1,
                'sequence_B': seq2,
                'nonstr_label': nonstr_label,
                'str_label': str_label,
                'mean_pLDDT': round(mean_plddt, 3), # average pLDDT of the best template
                'best_pLDDT': round(best_plddt, 3), # best pLDDT of the best template
                'mean_pAE': round(mean_pae, 3),     # mean pAE of the PAE matrix
                'mean_interface_pLDDT': round(mean_IF_pLDDT, 3), # mean value of the pLDDT values of the interface residues within the best template
                'best_interface_pLDDT': round(np.max(plddt_lst), 3), # best interface pLDDT out of interface residues within the best template
                'mean_interface_pAE': round(np.mean(avgif_pae), 3), # mean PAE values of the interface PAE from chain A and chain B
                'pDockQ': pDockQ,
                'mean_pDockQ2': round(mean_pdockq2, 3), # mean pDockQ2 value of both chains
                'best_pDockQ2': round(best_pdockq2, 3), # best pDockQ2 value out of both chains
                'mean_LIS': round(mean_lis_score, 3), # mean LIS value of the best prediction model
                'best_LIS': round(best_lis_score, 3),
                'mean_LIA': round(mean_lia_score,3), 
                'best_LIA': round(best_lia_score, 3),
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
    metrics_df = pd.DataFrame(columns=['mean_pLDDT', 'best_pLDDT', 'mean_pAE', 'mean_interface_pLDDT', 'best_interface_pLDDT', 'mean_interface_pAE', 'pDockQ', 'mean_pDockQ2', 'best_pDockQ2', 'mean_LIS', 'best_LIS', 'mean_LIA', 'best_LIA', 'ipTM', 'pTM', 'ranking_score'])


    metrics_df = process_zip(zip_folder, output_dir, metrics_df, MISSING_PPI_LABEL)

    return metrics_df
