import os

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






