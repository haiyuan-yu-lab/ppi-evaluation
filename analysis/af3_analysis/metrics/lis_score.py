# Modified from https://github.com/flyark/AFM-LIS/ release

import numpy as np

def reverse_and_scale_matrix(matrix: np.ndarray, pae_cutoff: float = 12.0) -> np.ndarray:
    """
    Scale the values in the matrix such that:
    0 becomes 1, pae_cutoff becomes 0, and values greater than pae_cutoff are also 0.
    
    Args:
    - matrix (np.ndarray): Input numpy matrix.
    - pae_cutoff (float): Threshold above which values become 0.
    
    Returns:
    - np.ndarray: Transformed matrix.
    """
    
    # Scale the values to [0, 1] for values between 0 and cutoff
    scaled_matrix = (pae_cutoff - matrix) / pae_cutoff
    scaled_matrix = np.clip(scaled_matrix, 0, 1)  # Ensures values are between 0 and 1
    
    return scaled_matrix


def compute_lis(pae, protein_a_len, pae_cutoff=12, print_results=False):

    thresholded_pae = np.where(pae < pae_cutoff, 1, 0)

    # Calculate the interaction amino acid numbers
    local_interaction_protein_a = np.count_nonzero(thresholded_pae[:protein_a_len, :protein_a_len])
    local_interaction_protein_b = np.count_nonzero(thresholded_pae[protein_a_len:, protein_a_len:])
    local_interaction_interface_1 = np.count_nonzero(thresholded_pae[:protein_a_len, protein_a_len:])
    local_interaction_interface_2 = np.count_nonzero(thresholded_pae[protein_a_len:, :protein_a_len])
    local_interaction_interface_avg = (
        local_interaction_interface_1 + local_interaction_interface_2
    )

    # Calculate average thresholded_pae for each region
    # average_thresholded_protein_a = thresholded_pae[:protein_a_len,:protein_a_len].mean() * 100
    # average_thresholded_protein_b = thresholded_pae[protein_a_len:,protein_a_len:].mean() * 100
    # average_thresholded_interaction1 = thresholded_pae[:protein_a_len,protein_a_len:].mean() * 100
    # average_thresholded_interaction2 = thresholded_pae[protein_a_len:,:protein_a_len].mean() * 100
    # average_thresholded_interaction_total = (average_thresholded_interaction1 + average_thresholded_interaction2) / 2
    
    pae_protein_a = np.mean( pae[:protein_a_len,:protein_a_len] )
    pae_protein_b = np.mean( pae[protein_a_len:,protein_a_len:] )
    pae_interaction1 = np.mean(pae[:protein_a_len,protein_a_len:])
    pae_interaction2 = np.mean(pae[protein_a_len:,:protein_a_len])
    pae_interaction_total = ( pae_interaction1 + pae_interaction2 ) / 2

    # For pae_A
    selected_values_protein_a = pae[:protein_a_len, :protein_a_len][thresholded_pae[:protein_a_len, :protein_a_len] == 1]
    average_selected_protein_a = np.mean(selected_values_protein_a)

    # For pae_B
    selected_values_protein_b = pae[protein_a_len:, protein_a_len:][thresholded_pae[protein_a_len:, protein_a_len:] == 1]
    average_selected_protein_b = np.mean(selected_values_protein_b)

    # For pae_interaction1
    selected_values_interaction1 = pae[:protein_a_len, protein_a_len:][thresholded_pae[:protein_a_len, protein_a_len:] == 1]
    average_selected_interaction1 = np.mean(selected_values_interaction1) if selected_values_interaction1.size > 0 else pae_cutoff

    # For pae_interaction2
    selected_values_interaction2 = pae[protein_a_len:, :protein_a_len][thresholded_pae[protein_a_len:, :protein_a_len] == 1]
    average_selected_interaction2 = np.mean(selected_values_interaction2) if selected_values_interaction2.size > 0 else pae_cutoff

    # For pae_interaction_total
    average_selected_interaction_total = (average_selected_interaction1 + average_selected_interaction2) / 2

    # At this point, plddt_data and pae_data dictionaries will have the extracted data
    # print_results = False
    if print_results:
        # Print the total results
        print("Total pae_A : {:.2f}".format(pae_protein_a))
        print("Total pae_B : {:.2f}".format(pae_protein_b))
        print("Total pae_i_1 : {:.2f}".format(pae_interaction1))
        print("Total pae_i_2 : {:.2f}".format(pae_interaction2))
        print("Total pae_i_avg : {:.2f}".format(pae_interaction_total))

        # Print the local results
        print("Local pae_A : {:.2f}".format(average_selected_protein_a))
        print("Local pae_B : {:.2f}".format(average_selected_protein_b))
        print("Local pae_i_1 : {:.2f}".format(average_selected_interaction1))
        print("Local pae_i_2 : {:.2f}".format(average_selected_interaction2))
        print("Local pae_i_avg : {:.2f}".format(average_selected_interaction_total))

        # Print the >PAE-cutoff area
        print("Local interaction area (Protein A):", local_interaction_protein_a)
        print("Local interaction area (Protein B):", local_interaction_protein_b)
        print("Local interaction area (Interaction 1):", local_interaction_interface_1)
        print("Local interaction area (Interaction 2):", local_interaction_interface_2)
        print("Total Interaction area (Interface):", local_interaction_interface_avg)


    # Transform the pae matrix
    scaled_pae = reverse_and_scale_matrix(pae, pae_cutoff)

    # For local interaction score for protein_a
    selected_values_protein_a = scaled_pae[:protein_a_len, :protein_a_len][thresholded_pae[:protein_a_len, :protein_a_len] == 1]
    # average_selected_protein_a_score = np.mean(selected_values_protein_a)

    # For local interaction score for protein_b
    selected_values_protein_b = scaled_pae[protein_a_len:, protein_a_len:][thresholded_pae[protein_a_len:, protein_a_len:] == 1]
    # average_selected_protein_b_score = np.mean(selected_values_protein_b)

    # For local interaction score1
    selected_values_interaction1_score = scaled_pae[:protein_a_len, protein_a_len:][thresholded_pae[:protein_a_len, protein_a_len:] == 1]
    average_selected_interaction1_score = np.mean(selected_values_interaction1_score) if selected_values_interaction1_score.size > 0 else 0

    # For local interaction score2
    selected_values_interaction2_score = scaled_pae[protein_a_len:, :protein_a_len][thresholded_pae[protein_a_len:, :protein_a_len] == 1]
    average_selected_interaction2_score = np.mean(selected_values_interaction2_score) if selected_values_interaction2_score.size > 0 else 0

    # For average local interaction score
    average_selected_interaction_total_score = (average_selected_interaction1_score + average_selected_interaction2_score) / 2

    return {'LIS': round(average_selected_interaction_total_score, 3), # Local Interaction Score (LIS)
            'LIA': local_interaction_interface_avg, # Local Interaction Area (LIA)
            'inter_pae': round(pae_interaction_total, 3),  # inter-chain pAE
            'inter_pae_select': round(average_selected_interaction_total, 3)  # inter-chain pAE below threshold (value equals threshold if all above cutoff value)
            }