import sys
import os

# Add project root to sys.path
PROJECT_ROOT = os.path.expanduser("~/ppi-evaluation")
sys.path.append(PROJECT_ROOT)


# Analysis path
AF3_ANALYSIS_PATH = os.path.join(PROJECT_ROOT, "analysis/af3_analysis")
AFM_ANALYSIS_PATH = os.path.join(PROJECT_ROOT, "analysis/afm_analysis")

# Output Paths
AF3_OUTPUT_DIR = os.path.join(AF3_ANALYSIS_PATH, "output")
AFM_OUTPUT_DIR = os.path.join(AFM_ANALYSIS_PATH, "output")

# Results Paths
AF3_RESULTS_CSV = os.path.join(AF3_OUTPUT_DIR, "af3_results.csv")
AFM_RESULTS_CSV = os.path.join(AFM_OUTPUT_DIR, "afm_results.csv")

# Input path
FASTA_PATH = "/home/yl986/data/protein_interaction/input_fasta"

# AF3 prediction folder
AF3_PRED_FOLDER = "/share/yu/ppi_pred/af3/af3-results"
AFM_PRED_FOLDER = "/home/yl986/data/afm_results"

# Visualization Colors
COLORS = {
    "nonstruct_neg": "#EF4035",
    "nonstruct_pos": "#6EB43F",
    "struct_neg": "#F8981D",
    "struct_pos": "#006699"
}

# Feature Columns
FEATURE_COLUMNS = ['LIS', 'ipTM', 'pTM', 'pLDDT', 'pDockQ', 'pAE', 'pDockQ2', 'ranking_score']

# Logging Utility
def log_message(message):
    print(f"[LOG]: {message}")



