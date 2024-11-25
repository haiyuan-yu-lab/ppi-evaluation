from pathlib import Path
from typing import Dict, Tuple
from Bio.PDB import MMCIFParser, PDBParser
import Bio
import logging
import json
import pickle
import warnings
import re
import numpy as np

"""
Protein structure data parsers from PIONEER evaluation pipeline: https://github.com/haiyuan-yu-lab/pioneer2-eval/tree/main
"""

log = logging.getLogger(__name__)


def parse_fasta(fasta_file: Path,
                uniprot_header=True,
                skip_metadata=False) -> Dict:
    """
    Parses a fasta file and produces a dictionary that maps the accession ID
    to the sequence and other metadata if available.

    Parameters
    ----------
    fasta_file : Path
        Path to the FASTA file
    uniprot_header : bool, default True
        If true, parses the Protein ID line using the following header as
        defined by UniProt. "UniqueIdentifier" will be used as the accession ID
        and all other elements will be added as metadata to each element

        if False, everything after ">" will be used as the accession ID, and
        the only metadata will be the sequence
    skip_metadata : bool, default False
        If true, only "sequence" will be included in the resulting dictionary.

    Returns
    -------
    Dict
        A dictionary with the following structure:
        {
            "accession_id": {
                "sequence": "...", # only this one is guaranteed to be included
                "db": "..."
                "EntryName": "..."
                ...
            }
        }
    """
    assert fasta_file.is_file()
    uniprot_re = None
    if uniprot_header:
        regex = (r"^>(?P<db>[a-zA-Z]+)\|(?P<UniqueIdentifier>\w+)\|"
                 r"(?P<EntryName>\w+)\s(?P<ProteinName>.*)\s"
                 r"OS=(?P<OrganismName>.*)\sOX=(?P<OrganismIdentifier>\w*)\s"
                 r"(?:GN=(?P<GeneName>.*)\s)?PE=(?P<ProteinExistence>.*)\s"
                 r"SV=(?P<SequenceVersion>\w+).*$")
        uniprot_re = re.compile(regex)

    sequences = {}
    metadata = {}
    curr_seq = ""
    curr_acc = ""
    with fasta_file.open() as f:
        for line in f:
            if line.startswith(">"):
                if curr_seq != "":
                    sequences[curr_acc] = {
                        "sequence": curr_seq
                    }
                    if not skip_metadata:
                        for m, v in metadata.items():
                            sequences[curr_acc][m] = v
                        metadata = {}
                    curr_seq = ""
                if uniprot_header:
                    match = uniprot_re.match(line)
                    if match:
                        metadata = match.groupdict()
                        curr_acc = metadata["UniqueIdentifier"]
                    else:
                        warnings.warn("Could not parse header, defaulting to"
                                      f"simple header for this entry {line}")
                        curr_acc = line[1:].split()[0]
                else:
                    curr_acc = line[1:].split()[0]
            else:
                curr_seq += line.strip()
        if curr_seq != "":
            sequences[curr_acc] = {
                "sequence": curr_seq
            }
            if not skip_metadata:
                for m, v in metadata.items():
                    sequences[curr_acc][m] = v
                metadata = {}
            curr_seq = ""

    return sequences


def _load_af3_files(
        res_dir: Path,
        model_idx: int) -> Tuple[Bio.PDB.Structure.Structure,
                                 Dict,
                                 Dict]:
    """
    Loads an AF3 model into memory

    Parameters
    ----------
    res_dir : Path
        Path to the directory containing .cif and .json files, downloaded from
        AF3 server
    model_idx : int
        0-based index of the models

    Returns
    -------
    structure : Biopython structure
    full_data : dictionaryk
        parsed from the *_full_data_<model_idx>.json
    summary : dictionary
        parsed from the *_summary_confidences_<model_idx>.json
    """
    options = list(res_dir.glob(f"*_model_{model_idx}.cif"))
    if len(options) != 1:
        raise ValueError(
            f"Model {model_idx} has {len(options)} cif candidates")
    cif_file = options[0]
    options = list(res_dir.glob(f"*_full_data_{model_idx}.json"))
    if len(options) != 1:
        raise ValueError(
            f"Model {model_idx} has {len(options)} full_data candidates")
    full_data_file = options[0]
    options = list(res_dir.glob(f"*_summary_confidences_{model_idx}.json"))
    if len(options) != 1:
        raise ValueError(
            f"Model {model_idx} has {len(options)} summary candidates")
    summary_file = options[0]
    parser = MMCIFParser()
    structure = parser.get_structure("AlphaFold", cif_file)
    full_data = json.load(full_data_file.open())
    summary = json.load(summary_file.open())
    return structure, full_data, summary


def parse_af3_result(res_dir: Path, load_all_models=False) -> Dict:
    """
    Parses the output folder from AlphaFold3 server, and returns a dictionary
    with more convenient navigability for per-residue analysis

    Parameters
    ----------
    res_dir : Path
        Path to the directory containing the .cif and .json files from AF3.
        Names are assumed to be unaltered.
    load_all_models : bool, default False
        If true, loads the 5 AF3 models into the dictionary. If false, it only
        loads the first (0th) model.

    Returns
    -------
    dictionary
        Dictionary with the following structure:
            <model_index>:{
                <chain_id>:{
                    <residue_id>:{
                        "plddt":[plddt values of all atoms],
                        "contact_probs": {<residue_id>: prob},
                        "pae": {<residue_id>: pae},
                    },
                    "chain_iptm": float,
                    "chain_ptm": float,
                    "chain_pair_iptm": {<chain_id>: float}
                    "chain_pair_pae_min": {<chain_id>: float}
                },
                "summary_confidences": { the same as in the json file
                                         unless related to a single chain }
            },

    Notes
    -----
    * Tokens are assumed to be amino acids
    * AF3 is assumed to produce 5 models
    """
    qty_models = 5 if load_all_models else 1
    result = {}
    for m in range(qty_models):
        structure, full_data, summary = _load_af3_files(res_dir, model_idx=m)
        # chain_plddt = {}
        # for , plddt in full_data["atom_plddts"]:
        #     cn = full_data["atom_chain_ids"][i]
        #     if cn not in chain_plddt:
        #         chain_plddt[cn] = []
        #     chain_plddt[cn].append(plddt)
        result[m] = {}
        model_data = result[m]
        for model in structure:
            chain_ent_to_name = {}
            for i, c in enumerate(model.get_chains()):
                chain_ent_to_name[i] = c.get_full_id()[2]
            for k in [
                "fraction_disordered",
                "has_clash",
                "iptm",
                "num_recycles",
                "ptm",
                "ranking_score"
            ]:
                model_data[k] = summary[k]
            
            chain_and_residue_to_token = {}
            token_to_chain_and_residue = {}
            for token, res_id in enumerate(full_data["token_res_ids"]):
                chain_id = full_data["token_chain_ids"][token]
                chain_and_residue_to_token[chain_id, res_id] = token
                token_to_chain_and_residue[token] = chain_id, res_id
                
            for chain in model:
                _, entity_id, chain_name = chain.get_full_id()
                model_data[chain_name] = {}
                chain_data = model_data[chain_name]
                chain_data["chain_iptm"] = summary["chain_iptm"][entity_id]
                chain_data["chain_ptm"] = summary["chain_ptm"][entity_id]
                chain_data["chain_pair_iptm"] = {
                    chain_ent_to_name[ent]: iptm
                    for ent, iptm in enumerate(
                        summary["chain_pair_iptm"][entity_id])
                }
                chain_data["chain_pair_pae_min"] = {
                    chain_ent_to_name[ent]: iptm
                    for ent, iptm in enumerate(
                        summary["chain_pair_pae_min"][entity_id])
                }
                # chain_and_residue_to_token = {}
                # token_to_chain_and_residue = {}
                # for token, res_id in enumerate(full_data["token_res_ids"]):
                #     chain_id = full_data["token_chain_ids"][token]
                #     chain_and_residue_to_token[chain_id, res_id] = token
                #     token_to_chain_and_residue[token] = chain_id, res_id
                chain_res = []
                for atom_idx, atom in enumerate(chain.get_atoms()):
                    _, res_id, _ = atom.parent.get_id()
                    if res_id not in chain_data:
                        chain_data[res_id] = {}
                        chain_data[res_id]["plddt"] = []
                        chain_res.append(res_id)
                        token_id = chain_and_residue_to_token[chain_name, res_id]
                        chain_data[res_id]["contact_probs"] = {
                            token_to_chain_and_residue[ti]: prob
                            for ti, prob in enumerate(
                                full_data["contact_probs"][token_id])
                        }
                        chain_data[res_id]["pae"] = {
                            full_data["token_res_ids"][ti]: prob
                            for ti, prob in enumerate(
                                full_data["pae"][token_id])
                        }
                    chain_data[res_id]["plddt"].append(
                        full_data["atom_plddts"][atom_idx]
                    )
                chain_data['residue_plddt'] = [np.mean(chain_data[res_id]['plddt']) for res_id in chain_res]

    return result


def _load_afm_result(afm_dir: Path, lite: bool) -> Tuple[Bio.PDB.Structure.Structure,
                                                         Dict]:
    """
    Loads an AFM model into memory

    Parameters
    ----------
    res_dir : Path
        Path to the directory containing .cif and .json files, downloaded from
        AF3 server
    lite : bool
        If true, the pickle file is ignored, and the result data is ignored, and
        the ranking_debug.json file is used instead

    Returns
    -------
    structure : Biopython structure
    result_data : dictionary
        parsed from the result_model_*_multimer_*.pkl file or the ranking debug json
    """
    ranked_0_file = afm_dir / "ranked_0.pdb"
    assert ranked_0_file.exists(), f"No rank 0 pdb in {afm_dir}"
    ranking_debug_file = afm_dir / "ranking_debug.json"
    parser = PDBParser()
    struct = parser.get_structure(afm_dir.name, ranked_0_file.open())
    assert ranking_debug_file.exists(), f"No rank debug file in {afm_dir}"
    ranking_debug = json.load(ranking_debug_file.open())
    r0 = ranking_debug["order"][0]
    result_data = {}
    if lite:
        result_data["ranking_confidence"] = float(ranking_debug["iptm+ptm"][r0])
    else:
        result_file = afm_dir / f'result_{r0}.pkl'
        assert result_file.exists(), f"No result pickle file in {afm_dir}"
        result_data = pickle.load(result_file.open("rb"))
    return struct, result_data


def parse_afm_result(afm_dir: Path, lite: bool = False) -> Dict:
    """
    Parses the output folder from AlphaFold-multimer, and returns a dictionary
    with more convenient navigability for per-residue analysis

    Parameters
    ----------
    res_dir : Path
        Path to the directory containing the .cif and .json files from AF3.
        Names are assumed to be unaltered.
    lite : bool, default False
        If true, information from the pickle file is not extracted, and no
        chain specific information is included

    Returns
    -------
    dictionary
        Dictionary with the following structure:
            <chain_id>:{
                <residue_id>:{
                    "plddt": float
                },
            },
            "iptm": float,
            "ptm": float,
            "ranking_confidence": float,

    Notes
    -----
    This will load only the results at the top ranking, and not all models.
    Not all keys in the result data are loaded.

    If `lite` is True, the dictionary contains only the "ranking_confidence" key
    """
    try:
        struct, result_data = _load_afm_result(afm_dir, lite=lite)
    except AssertionError as e:
        print(e)
        return
    result = {}
    if lite:
        result["ranking_confidence"] = float(result_data["ranking_confidence"])
    else:
        result["iptm"] = float(result_data["iptm"])
        result["ptm"] = float(result_data["ptm"])
        result["ranking_confidence"] = float(result_data["ranking_confidence"])
        cur_res_idx = 0
        for chain in struct.get_chains():
            result[chain.get_id()] = {}
            chain_data = result[chain.get_id()]
            for res_id, residue in enumerate(chain.get_residues(), start=1):
                if "plddt" not in chain_data:
                    chain_data["plddt"] = {}
                chain_data["plddt"][res_id] = float(result_data["plddt"][cur_res_idx])
                cur_res_idx += 1
    return result


# def parse_masif_integrated_file(filename: Path) -> Dict:
#     """
#     Loads a mapped MaSIF prediction

#     Parameters
#     ----------
#     filename : Path
#         Path to the mapped and integrated MaSIF prediction

#     Returns
#     -------
#     dictionary
#         Dictionary with the following structure:
#             <chain_id>:{
#                 <residue_id>: [float, float, ...]
#             }

#     Notes
#     -----
#     Each residue holds all scores from associated MaSIF vertices, and it is up
#     to the user to aggregate these however they think it's most useful.
#     """
#     assert filename.is_file(), f"The provided file does not exist: {filename}"
#     header = True
#     results = {}
#     with filename.open() as masif_file:
#         for line in masif_file:
#             if header:
#                 header = False
#                 continue
#             _, _, _, _, entity_id, _, pred = line.strip().split()
#             pred = float(pred)
#             match entity_id.split(":"):
#                 case chain_id, _, residue_id:
#                     if chain_id not in results:
#                         results[chain_id] = {}
#                     if residue_id not in results[chain_id]:
#                         results[chain_id][residue_id] = []
#                     results[chain_id][residue_id].append(pred)
#                 case chain_id, _, residue_id, insertion:
#                     if chain_id not in results:
#                         results[chain_id] = {}
#                     res_id = f"{residue_id}{insertion}"
#                     if res_id not in results[chain_id]:
#                         results[chain_id][res_id] = []
#                     results[chain_id][res_id].append(pred)
#                 case _:
#                     print("problem with line", entity_id, line)

#     return results
