"""
pDockQ2 score calculation from https://gitlab.com/ElofssonLab/FoldDock
"""

from Bio.PDB import PDBIO
from Bio.PDB.PDBParser import PDBParser
from Bio.PDB.Selection import unfold_entities

import numpy as np
import sys
import os
import argparse
import pickle
import itertools
import pandas as pd
from scipy.optimize import curve_fit


parser = argparse.ArgumentParser(
    description='''Calculate chain_level pDockQ_i. ''')
parser.add_argument('-pkl', nargs=1, type=str,
                    required=True, help='Input pickle file.')
parser.add_argument('-pdb', nargs=1, type=str,
                    required=True, help='Input pdb file.')
parser.add_argument(
    "-dist", help="maximum distance of a contact", nargs='?', type=int, default=8)


def retrieve_iplddt(structure, chain1, chain2_lst, max_dist):
    # generate a dict to save IF_res_id
    chain_lst = list(chain1) + chain2_lst

    i_plddt_lst = []
    contact_chain_lst = []
    for res1 in structure[0][chain1]:
        for chain2 in chain2_lst:
            count = 0
            for res2 in structure[0][chain2]:
                if res1.has_id('CA') and res2.has_id('CA'):
                    dis = abs(res1['CA']-res2['CA'])
                    # add criteria to filter out disorder res
                    if dis <= max_dist:
                        i_plddt_lst.append(res1['CA'].get_bfactor())
                        count += 1

                elif res1.has_id('CB') and res2.has_id('CB'):
                    dis = abs(res1['CB']-res2['CB'])
                    if dis <= max_dist:
                        i_plddt_lst.append(res1['CB'].get_bfactor())
                        count += 1
            if count > 0:
                contact_chain_lst.append(chain2)
    contact_chain_lst = sorted(list(set(contact_chain_lst)))

    if len(i_plddt_lst) > 0:
        i_plddt_mean = np.mean(i_plddt_lst)
    else:
        i_plddt_mean = 0

    return i_plddt_mean, contact_chain_lst


def retrieve_ipae_inter(structure, pae, contact_lst, dist_cutoff=8):
    """
    Retrieve interface pAE (inter-chain only)
    """
    # contact_lst:the chain list that have an interface with each chain. For eg, a tetramer with A,B,C,D chains and A/B A/C B/D C/D interfaces,
    # contact_lst would be [['B','C'],['A','D'],['A','D'],['B','C']]

    chain_lst = [x.id for x in structure[0]]
    seqlen = [len(x) for x in structure[0]]
    ifpae_avg = []
    d = 10
    for ch1_idx in range(len(chain_lst)):
        # extract x axis range from the PAE matrix
        idx = chain_lst.index(chain_lst[ch1_idx])
        chain1_start = sum(seqlen[:idx])
        chain1_end = chain1_start+seqlen[idx]
        ifpae_col = []
        # for each chain that shares an interface with chain1, retrieve the PAE matrix for the specific part.
        for contact_ch in contact_lst[ch1_idx]:
            index = chain_lst.index(contact_ch)
            chain2_start = sum(seqlen[:index])
            chain2_end = chain2_start + seqlen[index]
            remain_pae = pae[chain1_start:chain1_end, chain2_start:chain2_end]  # inter-chain pAE
            # print(contact_ch, ch1_sta, ch1_end, ch_sta, ch_end)

            # get avg PAE values for the interfaces for chain 1
            mat_x = -1
            for res1 in structure[0][chain_lst[ch1_idx]]:
                mat_x += 1
                mat_y = -1
                for res2 in structure[0][contact_ch]:
                    mat_y += 1
                    if res1['CA'] - res2['CA'] <= dist_cutoff:
                        ifpae_col.append(remain_pae[mat_x, mat_y])
        # normalize by d(10A) first and then get the average
        if not ifpae_col:
            ifpae_avg.append(0)
        else:
            norm_ipae = np.mean(1 / (1 + (np.array(ifpae_col) / d) ** 2))
            ifpae_avg.append(norm_ipae)

    return ifpae_avg


def calc_pmidockq(ifpae_norm, ifplddt):
    df = pd.DataFrame()
    df['ifpae_norm'] = ifpae_norm
    df['ifplddt'] = ifplddt
    df['prot'] = df.ifpae_norm*df.ifplddt
    fitpopt = [1.31034849e+00, 8.47326239e+01, 7.47157696e-02,
               5.01886443e-03]  # from orignal fit function
    df['pmidockq'] = sigmoid(df.prot.values, *fitpopt)

    return df


def sigmoid(x, L, x0, k, b):
    y = L / (1 + np.exp(-k*(x-x0)))+b
    return (y)


def fit_newscore(df, column):

    testdf = df[df[column] > 0]

    colval = testdf[column].values
    dockq = testdf.DockQ.values
    xdata = colval[np.argsort(colval)]
    ydata = dockq[np.argsort(dockq)]

    # this is an mandatory initial guess
    p0 = [max(ydata), np.median(xdata), 1, min(ydata)]
    # method='dogbox', maxfev=50000)
    popt, pcov = curve_fit(sigmoid, xdata, ydata, p0)

#    tiny=1.e-20
#    print('L=',np.round(popt[0],3),'x0=',np.round(popt[1],3), 'k=',np.round(popt[2],3), 'b=',np.round(popt[3],3))

    # plotting
#    x_pmiDockQ = testdf[column].values
#    x_pmiDockQ = x_pmiDockQ[np.argsort(x_pmiDockQ)]
#    y_pmiDockQ = sigmoid(x_pmiDockQ, *popt)
#    print("Average error for sigmoid fit is ", np.average(np.absolute(y_pmiDockQ-ydata)))

    # sns.kdeplot(data=df,x=column,y='DockQ',kde=True,levels=5,fill=True, alpha=0.8, cut=0)
#    sns.scatterplot(data=df,x=column,y='DockQ', hue='class')
#    plt.legend([],[], frameon=False)

#    plt.plot(x_pmiDockQ, y_pmiDockQ,label='fit',color='k',linewidth=2)
    return popt


def calc_pdockq2(structure, pae_mat, dist_cutoff=8):
    """
    re-organized pDockQ2 computation (YL)
    """
    fitpopt = [1.31034849e+00, 8.47326239e+01, 7.47157696e-02, 5.01886443e-03]  # from orignal fit function

    chains = []
    for chain in structure[0]:
        chains.append(chain.id)

    remain_contact_lst = []
    # retrieve interface plDDT at chain-level
    iplddt_lst = []
    for idx in range(len(chains)):
        chain2_lst = list(set(chains)-set(chains[idx]))
        chain_i_plddt, contact_lst = retrieve_iplddt(structure, chains[idx], chain2_lst, dist_cutoff)
        iplddt_lst.append(chain_i_plddt)
        remain_contact_lst.append(contact_lst)

    i_pae_lst = retrieve_ipae_inter(structure, pae_mat, remain_contact_lst, dist_cutoff)
    chain_pdockq2 = pdockq2_formula(i_pae_lst, iplddt_lst, *fitpopt)

    return {'iplddt_mean': np.mean(iplddt_lst),  # interface pLDDT
            'iplddt_chain_a': iplddt_lst[0],
            'iplddt_chain_b': iplddt_lst[1],
            'ipae_mean': np.mean(i_pae_lst),  # inter-chain pAE on interface
            'ipae_chain_b': i_pae_lst[1],
            'ipae_chain_a': i_pae_lst[0],
            'pdockq2_mean': np.mean(chain_pdockq2)}


def pdockq2_formula(i_pae, i_plddt, L, x0, k, b):
    if not isinstance(i_pae, np.ndarray):
        i_pae = np.array(i_pae)
    if not isinstance(i_plddt, np.ndarray):
        i_plddt = np.array(i_plddt)
    
    return L / (1 + np.exp(-k * (i_pae * i_plddt - x0))) + b


def main():
    args = parser.parse_args()
    pdbp = PDBParser(QUIET=True)
    iopdb = PDBIO()

    structure = pdbp.get_structure('', args.pdb[0])
    chains = []
    for chain in structure[0]:
        chains.append(chain.id)

    remain_contact_lst = []
    # retrieve interface plDDT at chain-level
    plddt_lst = []
    for idx in range(len(chains)):
        chain2_lst = list(set(chains)-set(chains[idx]))
        i_plddt, contact_lst = retrieve_iplddt(structure, chains[idx], chain2_lst, args.dist)
        plddt_lst.append(i_plddt)
        remain_contact_lst.append(contact_lst)

    # retrieve interface PAE at chain-level
    with open(args.pkl[0], 'rb') as f:
        data = pickle.load(f)

    avgif_pae = retrieve_ipae_inter(structure, data['predicted_aligned_error'], remain_contact_lst, args.dist)
    # calculate pmiDockQ

    res = calc_pmidockq(avgif_pae, plddt_lst)

    # print output
    print('pDockQ_i is:')
    for ch in range(len(chains)):
        print(chains[ch]+' '+str(res['pmidockq'].tolist()[ch]))


if __name__ == '__main__':

    main()
