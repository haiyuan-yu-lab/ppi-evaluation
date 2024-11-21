import numpy as np
import os
from util import *
from Bio.PDB.MMCIFParser import MMCIFParser
from Bio import SeqIO
import pandas as pd



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



def retrieve_IFplddt(structure, chain1, chain2_lst, max_dist):
    ## generate a dict to save IF_res_id
    chain_lst = list(chain1) + chain2_lst

    ifplddt = []
    contact_chain_lst = []
    for res1 in structure[0][chain1]:
        for chain2 in chain2_lst:
            count = 0
            for res2 in structure[0][chain2]:
                if res1.has_id('CA') and res2.has_id('CA'):
                   dis = abs(res1['CA']-res2['CA'])
                   ## add criteria to filter out disorder res
                   if dis <= max_dist:
                      ifplddt.append(res1['CA'].get_bfactor())
                      count += 1

                elif res1.has_id('CB') and res2.has_id('CB'):
                   dis = abs(res1['CB']-res2['CB'])
                   if dis <= max_dist:
                      ifplddt.append(res1['CB'].get_bfactor())
                      count += 1
            if count > 0:
              contact_chain_lst.append(chain2)
    contact_chain_lst = sorted(list(set(contact_chain_lst)))   


    if len(ifplddt)>0:
       IF_plddt_avg = np.mean(ifplddt)
    else:
       IF_plddt_avg = 0

    return IF_plddt_avg, contact_chain_lst


def retrieve_IFPAEinter(structure, paeMat, contact_lst, max_dist):
    ## contact_lst:the chain list that have an interface with each chain. For eg, a tetramer with A,B,C,D chains and A/B A/C B/D C/D interfaces,
    ##             contact_lst would be [['B','C'],['A','D'],['A','D'],['B','C']]

 
    chain_lst = [x.id for x in structure[0]]
    seqlen = [len(x) for x in structure[0]]
    ifch1_col=[]
    ifch2_col=[]
    ch1_lst=[]
    ch2_lst=[]
    ifpae_avg = []
    d=10
    for ch1_idx in range(len(chain_lst)):
      ## extract x axis range from the PAE matrix
      idx = chain_lst.index(chain_lst[ch1_idx])
      ch1_sta=sum(seqlen[:idx])
      ch1_end=ch1_sta+seqlen[idx]
      ifpae_col = []   
      ## for each chain that shares an interface with chain1, retrieve the PAE matrix for the specific part.
      for contact_ch in contact_lst[ch1_idx]:
        index = chain_lst.index(contact_ch)
        ch_sta = sum(seqlen[:index])
        ch_end = ch_sta+seqlen[index]
        remain_paeMatrix = paeMat[ch1_sta:ch1_end,ch_sta:ch_end]
        #print(contact_ch, ch1_sta, ch1_end, ch_sta, ch_end)        

        ## get avg PAE values for the interfaces for chain 1
        mat_x = -1
        for res1 in structure[0][chain_lst[ch1_idx]]:
          mat_x += 1
          mat_y = -1
          for res2 in structure[0][contact_ch]:
              mat_y+=1
              if res1['CA'] - res2['CA'] <=max_dist:
                 ifpae_col.append(remain_paeMatrix[mat_x,mat_y])
      ## normalize by d(10A) first and then get the average
      if not ifpae_col:
        ifpae_avg.append(0)
      else:
        norm_if_interpae=np.mean(1/(1+(np.array(ifpae_col)/d)**2))
        ifpae_avg.append(norm_if_interpae)

    return ifpae_avg
    

def calc_pmidockq(ifpae_norm, ifplddt):
    df = pd.DataFrame()
    df['ifpae_norm'] = ifpae_norm
    df['ifplddt'] = ifplddt
    df['prot'] = df.ifpae_norm*df.ifplddt
    fitpopt = [1.31034849e+00, 8.47326239e+01, 7.47157696e-02, 5.01886443e-03]  
    df['pmidockq'] = sigmoid(df.prot.values, *fitpopt)

    return df

def sigmoid(x, L ,x0, k, b):
    y = L / (1 + np.exp(-k*(x-x0)))+b
    return (y)





##############################################################################################################################


# Function to assign structural and nonstructural labels for catergorizations.
def categorize_predictions(pred_protein_a, pred_protein_b):
    pred_protein_a = pred_protein_a.upper()
    pred_protein_b = pred_protein_b.upper()
    ppi = pred_protein_a + ':' + pred_protein_b

    label_data = pd.read_csv('/home/yl986/data/protein_interaction/results/af3_ppi_with_label.csv')

    for _, row in label_data.iterrows():
        ppi_ref = row['ppi']
        if ppi_ref == ppi:
            nonstr_label = row['non_struct_label']
            str_label = row['str_label']
            return nonstr_label, str_label

    # If no match is found, return None for both labels
    print(f"No match found for {ppi}. Returning None for labels.")
    return None, None


















