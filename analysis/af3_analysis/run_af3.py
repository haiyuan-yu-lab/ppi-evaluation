from dataset_preparation import *
from util import *

def main():
    print("Starting AlphaFold3 dataset preparation...")
    
    # This line should be deleted once we finish running newly added samples.
    af3_df = create_af3_dataset(AF3_PRED_FOLDER, AF3_OUTPUT_DIR)

    af3_df.to_csv(AF3_RESULTS_CSV, index=False)
    #######################################################################

    #######################  ANALYSIS #####################################

    # STEP 1: Show the distribution of AF3 predicted score features


    # STEP 2: Draw ROC-PR curves on AF3 features
    af3_df =  normalize_scores(af3_df)

    # Make sure that nonstructural datasets should maintain ~1:13 ratio of the number of positive and negative samples


    # Visualize ROC-PR curves

    print(f"Results saved to {AF3_RESULTS_CSV}")



if __name__ == "__main__":
    main()


