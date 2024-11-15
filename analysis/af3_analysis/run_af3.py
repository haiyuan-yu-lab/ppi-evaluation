from dataset_preparation import *
from util import *

def main():
    print("Starting AlphaFold3 dataset preparation...")
    af3_df = create_af3_dataset(AF3_PRED_FOLDER, AF3_OUTPUT_DIR)

    af3_df.to_csv(AF3_RESULTS_CSV, index=False)
    print(f"Results saved to {AF3_RESULTS_CSV}")



if __name__ == "__main__":
    main()


