from dataset_preparation import *
from util import *

def main():
    print("Starting AlphaFold-Multimer dataset preparation....")
    afm_df = create_afm_dataset(AFM_PRED_FOLDER)

    afm_df.to_csv(AFM_RESULTS_CSV, index=False)



if __name__ == "__main__":
    main()


