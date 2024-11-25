from dataset_preparation import *
from util import *
from analyze import *  


def main():


    # This line should be deleted once we finish running newly added samples.
    af3_df = create_af3_dataset(AF3_PRED_FOLDER, AF3_OUTPUT_DIR)  # AF3_PRED_FOLDER = '/share/yu/ppi-pred/af3/af3_results'

    af3_df.to_csv(AF3_RESULTS_CSV, index=False)
    #######################################################################

    # Normalize AF3 prediction scores that are out-of-bounds of [0, 1]
    #af3_df = normalize_scores(af3_df)

    # Check the number of samples in each category of the dataset
    print(f"total # samples: {len(af3_df)}")
    print(f"Structural Positives: {len(af3_df[af3_df['str_label'] == 1])}")
    print(f"Structural Negatives: {len(af3_df[af3_df['str_label'] == 0])}")
    print(f"Nonstructural Positives: {len(af3_df[af3_df['nonstr_label'] == 1])}")
    print(f"Nonstructural Negatives: {len(af3_df[af3_df['nonstr_label'] == 0])}")
    
    # Create structural & nonstructural datasets
    struct_df = af3_df[(af3_df['str_label'] == 1) | (af3_df['str_label'] == 0)]
    nonstruct_df = af3_df[(af3_df['nonstr_label'] == 1) | (af3_df['nonstr_label'] == 0)]

    struct_df['label'] = struct_df['str_label']
    nonstruct_df['label'] = nonstruct_df['nonstr_label']



    #######################  ANALYSIS #####################################

    # STEP 1: Show the distribution of AF3 predicted score features for each category
    plot_histograms(struct_df, "structural")
    plot_histograms(nonstruct_df, "nonstructural")

    return -1
    
    features = ['mean_pLDDT', 'best_pLDDT', 'mean_pAE', 'mean_interface_pLDDT', 'best_interface_pLDDT', 'mean_interface_pAE', 'pDockQ', 'mean_pDockQ2', 'best_pDockQ2', 'mean_LIS', 'ipTM', 'pTM', 'ranking_score']

    # STEP 2: Draw ROC-PR curves on AF3 features. Ensure to maintain ~1:13 ratio of the number of positive and negative samples for nonstructural set
    
    plot_feature_roc_curves(struct_df, "structural", features)
    plot_feature_roc_curves(nonstruct_df, "nonstructural", features)

    plot_feature_pr_curves(struct_df, "structural", features)
    plot_feature_pr_curves(nonstruct_df, "nonstructural", features)



    # Visualize ROC-PR curves

    print(f"Results saved to {AF3_RESULTS_CSV}")



if __name__ == "__main__":
    main()


