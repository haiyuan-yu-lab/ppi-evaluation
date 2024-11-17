import json
import os

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
