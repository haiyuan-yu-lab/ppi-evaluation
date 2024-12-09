import os, sys
from pathlib import Path
from typing import Dict, Tuple

import zipfile
import shutil
import json
import pickle
import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(__file__))
sys.path.append(os.path.abspath('..'))

from prot_data_parsers import _load_af3_files
from metrics.pdockq import *
from metrics.pdockq2 import calc_pdockq2
from metrics.lis_score import compute_lis


def read_cif(structure):
    '''
    Read a .cif file to contain all chains. (from Juheon)
    '''
    chain_coords, chain_plddt = {}, {}

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


def calc_af3_metrics_base(res_fpath, ppi, model_idx=0, ppi_sep=':'):
    if isinstance(res_fpath, str):
        res_fpath = Path(res_fpath)
    
    structure, full_data, summary_data = _load_af3_files(res_fpath, model_idx=model_idx)
    # plddt = np.mean(np.array(full_data['atom_plddts']))
    pae = np.array(full_data['pae'])  # residue level PAE
    mean_pae = np.mean(pae)

    iptm = summary_data.get('iptm', np.nan)
    ptm = summary_data.get('ptm', np.nan)
    ranking_confidence = summary_data.get('ranking_score', np.nan)

    chain_coords, chain_plddt = read_cif(structure)
    res_plddts = np.concatenate([chain_plddt[c] for c in chain_plddt.keys()])
    plddt_mean = np.mean(res_plddts)
    chain_lengths = {k: v.shape[0] for k, v in chain_coords.items()}
    chain_a = sorted(chain_lengths.keys())[0]
    protein_a_len = chain_lengths[chain_a]

    # Calculate pdockq
    pdockq, ppv = calc_pdockq(chain_coords, chain_plddt, t=8) # Distance threshold, set to 8
    # TODO: pDockQ2
    try:
        prot1, prot2 = ppi.split(ppi_sep, 1)
    except ValueError:
        print('Exception for', ppi)
        return {}

    result_dict = {
        'ppi': ':'.join(sorted([prot1, prot2])),
        'pDockQ': pdockq, 
        'PPV': round(ppv, 3),
        'prot1': prot1,
        'prot2': prot2,
        'ipTM': round(float(iptm), 3),
        'iptm_ptm': round(float(iptm*0.8 + ptm*0.2),3),
        'pTM': round(float(ptm), 3),
        'pLDDT': round(plddt_mean, 2),
        'pAE': round(mean_pae, 3),
        'ranking_confidence': ranking_confidence
    }
    
    # Compute LIS
    lis_res_dict = compute_lis(pae, protein_a_len=protein_a_len)

    result_dict.update(lis_res_dict)
    pdockq2_res_dict = calc_pdockq2(structure, pae)
    result_dict.update({k: round(v, 3) for k,v in pdockq2_res_dict.items()})

    return result_dict


def assign_ppi_label(df_ppi_raw, nonstr_pos, nonstr_ex, str_pos, ppi_in_pdb):
    df_ppi = df_ppi_raw.copy()
    df_ppi['ppi'] = df_ppi_raw['ppi'].apply(lambda x: ':'.join(sorted(x.split(':'))))  # make sure proteins are sorted
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


if __name__ == '__main__':
    raw_result_root = Path('/share/yu/ppi_pred/af3/af3-results/')
    data_root = Path('/home/yl986/data/protein_interaction')
    parsed_result_root = data_root / 'results'
    # result_cache_file = '/home/yl986/alphafold-2.3.2/logs/str_pred.log'
    output_cache_path = '/home/yl986/data/protein_interaction/results/af3_scores_20241204.csv'  # already parsed
    # output_path = '/home/yl986/data/protein_interaction/results/af3_scores_test.csv'
    output_path = '/home/yl986/data/protein_interaction/results/af3_scores_20241209.csv'
    af3_extract_log = '/home/yl986/data/protein_interaction/results/af3_zip_complete.log'

    OVERWRITE = False
    if os.path.exists(output_cache_path):
        df_complete = pd.read_csv(output_cache_path)
        complete_set = set(df_complete['ppi'].tolist())
        print(len(complete_set), 'pairs already processed')
    else:
        df_complete = None
        complete_set = set()
    
    if os.path.exists(af3_extract_log):
        with open(af3_extract_log) as f:
            complete_zips = f.read().splitlines()
    else:
        complete_zips = []

    # List all files in the directory
    f_list = list(raw_result_root.glob('fold*zip'))
    print(f'{len(f_list)} zipped files to process...')
    results_all = []
    for f_path in f_list:
        if not OVERWRITE and f_path.name in complete_zips:
            continue
        if f_path.stem.startswith('folds'):
            temp_batch_dir = Path('temp')
            with zipfile.ZipFile(f_path, 'r') as zip_ref:
                zip_ref.extractall(temp_batch_dir)
            for cur_path in temp_batch_dir.glob('*'):
                if cur_path.is_dir():
                    ppi = cur_path.stem.upper().replace('_', ':', 1)
                    if ppi in complete_set and not OVERWRITE:
                        complete_zips.append(f_path.name)
                        continue
                    cur_res_dict = calc_af3_metrics_base(cur_path, ppi)
                    if not cur_res_dict:
                        continue
                    results_all.append(cur_res_dict)
                    complete_set.add(ppi)
                    complete_zips.append(f_path.name)

            shutil.rmtree(temp_batch_dir)
        
        else:
            # Extract the directory name from the zip file name
            # base_name = os.path.splitext(f_path)[0]  # Remove the .zip extension
            temp_dir_path = Path(f_path.stem.replace("fold_", ""))   # Remove the "fold_" prefix

            ppi = temp_dir_path.stem.upper().replace('_', ':', 1)
            if ppi in complete_set and not OVERWRITE:
                complete_zips.append(f_path.name)
                continue

            # Unzip the file into the target directory
            with zipfile.ZipFile(f_path, 'r') as zip_ref:
                zip_ref.extractall(temp_dir_path)

            cur_res_dict = calc_af3_metrics_base(temp_dir_path, ppi)
            if not cur_res_dict:
                continue
            results_all.append(cur_res_dict)
            complete_set.add(ppi)
            complete_zips.append(f_path.name)
            # After operations, remove the directory
            shutil.rmtree(temp_dir_path)

        # if len(results_all) > 10:  # for debugging
        #     break
    print(f'{len(results_all)} interactions processed')
    
    with open(af3_extract_log, 'w') as f:  # write complete log
        f.write('\n'.join(sorted(set(complete_zips))))

    df_result = pd.DataFrame.from_dict(results_all)
    if not OVERWRITE and not isinstance(df_complete, type(None)):
        df_result = pd.concat([df_complete, df_result]).drop_duplicates(['ppi']).reset_index(drop=True)
    
    with open(data_root / 'parsed/ppi_label_reference.pkl', 'rb') as f:
        # reference dict for label assignment --- 
        # keys: "nonstr_pos", "nonstr_exclusion", "true_ppi_in_pdb", "all_pair_in_pdb"
        ppi_label_ref_dict = pickle.load(f)
    
    df_result_labeled = assign_ppi_label(df_result, ppi_label_ref_dict['nonstr_pos'], ppi_label_ref_dict['nonstr_exclusion'], 
                                 ppi_label_ref_dict['true_ppi_in_pdb'], ppi_label_ref_dict['all_pair_in_pdb'])
    print(df_result_labeled.groupby('str_label')['ppi'].nunique())
    print(df_result_labeled.groupby('non_struct_label')['ppi'].nunique())
    df_result_labeled['is_homo'] = (df_result_labeled['prot1'] == df_result_labeled['prot2'])
    df_result_labeled.to_csv(output_path, index=False)

    print("All operations completed successfully.")
