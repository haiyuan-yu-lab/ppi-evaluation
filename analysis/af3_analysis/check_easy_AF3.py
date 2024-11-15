from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio import SeqIO  
from Bio.PDB.MMCIFParser import MMCIFParser
from Bio import PDB

import subprocess
import os
import zipfile
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, precision_recall_curve, auc
from sklearn.model_selection import KFold
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split  
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
from collections import defaultdict
import random
import seaborn as sns
from sklearn.utils import resample
from sklearn.ensemble import RandomForestClassifier 
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import GradientBoostingClassifier, AdaBoostClassifier
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    precision_score, recall_score, f1_score, roc_auc_score, matthews_corrcoef,
    average_precision_score, make_scorer  
)
from sklearn.model_selection import StratifiedKFold, GridSearchCV, train_test_split
from sklearn.metrics import precision_recall_curve, auc

from sklearn.inspection import permutation_importance
from sklearn.feature_selection import RFE
from sklearn.decomposition import FactorAnalysis
from sklearn.linear_model import LogisticRegression
import shap
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.stats import pearsonr, spearmanr, kendalltau
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from Bio.Align import MultipleSeqAlignment

from sklearn.linear_model import LinearRegression

colors = {
    "nonstruct_neg": "#EF4035",
    "nonstruct_pos": "#6EB43F",
    "struct_neg": "#F8981D",
    "struct_pos": "#006699"
}


param_grids = {

    'Random Forest': {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2],
        'bootstrap': [True],
        'max_features': ['sqrt']
    },
        
    'SVM': {
        'C': [0.1, 1],
        'gamma': ['scale'],
        'kernel': ['rbf']
    },
        
    'MLP': {
        'hidden_layer_sizes': [(100,)],
        'activation': ['relu'],
        'solver': ['adam'],
        'alpha': [0.0001],
        'learning_rate': ['constant'],
        'max_iter': [200]
    }
}

classifiers = {
    'Random Forest': RandomForestClassifier(random_state=42),
    'SVM': SVC(probability=True, random_state=42),
    'MLP': MLPClassifier(max_iter=1000, random_state=42)
}


################################################### Feature Selection ###############################################
# Function to plot LASSO coefficient distributions with the improved style
def plot_lasso_coefficients(X_train, y_train, feature_names, alpha, dataset_name):
    """
    Fit LASSO and plot the distribution of LASSO coefficients with enhanced visualization.
    
    Args:
        X_train (pd.DataFrame): The training feature set.
        y_train (pd.Series): The target variable for the training set.
        feature_names (list): List of feature names to keep track of.
        alpha (float): Regularization strength for LASSO (default 0.01).
    """
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    # Initialize the Lasso model with L1 regularization
    lasso = Lasso(alpha=alpha, max_iter=10000)
    lasso.fit(X_train_scaled, y_train)
    
    # Get the coefficients of the trained LASSO model
    coef = lasso.coef_
    
    # Sorting coefficients and features for better visualization
    sorted_indices = np.argsort(coef)
    sorted_coef = coef[sorted_indices]
    sorted_features = np.array(feature_names)[sorted_indices]
    
    # Plot the coefficient distribution
    plt.figure(figsize=(10, 6))
    bars = plt.barh(sorted_features, sorted_coef, color=np.where(sorted_coef > 0, '#006699', 'red'))
    
    # Add the coefficient values next to the bars for clarity
    for bar in bars:
        width = bar.get_width()
        plt.text(width, bar.get_y() + bar.get_height()/2, f'{width:.2f}', va='center', ha='left' if width > 0 else 'right')

    # Customize the plot
    plt.title(f"LASSO Coefficients ({dataset_name})", fontsize=16)
    plt.xlabel("Coefficient Value", fontsize=14)
    plt.ylabel("Feature", fontsize=14)
    plt.grid(True, axis='x', linestyle='--', alpha=0.7)
    plt.tight_layout()

    # Save the plot as a file and show
    plt.savefig(f"AF3_easy_lasso_{dataset_name}.png")
    plt.show()



def balance_dataset(df, label_column='label'):
    """
    Balance the dataset by undersampling the majority class to match the minority class size.
    """
    positives = df[df[label_column] == 1]
    negatives = df[df[label_column] == 0]

    # Undersample the majority class to match the minority class
    if len(positives) > len(negatives):
        positives = resample(positives, replace=False, n_samples=len(negatives), random_state=42)
    else:
        negatives = resample(negatives, replace=False, n_samples=len(positives), random_state=42)

    return pd.concat([positives, negatives])


# Function to read benchmark datasets
def load_benchmark_data(path):
    struct_pos_df = pd.read_csv(os.path.join(path, 'struct_pos_benchmark.csv'))
    struct_neg_df = pd.read_csv(os.path.join(path, 'struct_neg_benchmark.csv'))
    nonstruct_pos_df = pd.read_csv(os.path.join(path, 'non_struct_pos_benchmark.csv'))
    nonstruct_neg_df = pd.read_csv(os.path.join(path, 'non_struct_neg_benchmark.csv'))
    return struct_pos_df, struct_neg_df, nonstruct_pos_df, nonstruct_neg_df



# Function to parse CIF file and extract chain coordinates and pLDDT values (for all atoms)
def read_cif(cif_file):
    '''
    Read a .cif file to contain all chains.
    '''
    chain_coords, chain_plddt = {}, {}
    try:
        parser = MMCIFParser(QUIET=True)
        structure = parser.get_structure("structure", cif_file)

        for model in structure:
            for chain in model:
                for residue in chain:
                    for atom in residue:
                        if atom.get_name() == 'CB' or (atom.get_name() == 'CA' and residue.get_resname() == 'GLY'):
                            if chain.id in chain_coords:
                                chain_coords[chain.id].append(atom.get_coord())
                                chain_plddt[chain.id].append(atom.get_bfactor())  # Assuming pLDDT values are stored in the B-factor field
                            else:
                                chain_coords[chain.id] = [atom.get_coord()]
                                chain_plddt[chain.id] = [atom.get_bfactor()]

        # Convert to arrays
        for chain in chain_coords:
            chain_coords[chain] = np.array(chain_coords[chain])
            chain_plddt[chain] = np.array(chain_plddt[chain])

        return chain_coords, chain_plddt

    except Exception as e:
        print(f"Error parsing CIF file {cif_file}: {e}")
        return None, None


# Function to assign structural and nonstructural labels for catergorizations.
def categorize_predictions(pred_protein_a, pred_protein_b):
    pred_protein_a = pred_protein_a.upper()
    pred_protein_b = pred_protein_b.upper()
    ppi = pred_protein_a + ':' + pred_protein_b
    print(ppi)
     
    label_data = pd.read_csv('/home/yl986/data/protein_interaction/results/af3_ppi_with_label.csv')
    
    for _, row in label_data.iterrows():
        ppi_ref = row['ppi']
        if ppi_ref == ppi:
            nonstr_label = row['nonstr_label']
            str_label = row['str_label']
            return nonstr_label, str_label

# Function to reverse and scale matrix
def reverse_and_scale_matrix(matrix: np.ndarray, pae_cutoff: float = 12.0) -> np.ndarray:
    scaled_matrix = (pae_cutoff - matrix) / pae_cutoff
    return np.clip(scaled_matrix, 0, 1)


# Function to calculate pDockQ
def calc_pdockq(chain_coords, chain_plddt, t):

    # Get coords and plddt per chain
    ch1, ch2 = [*chain_coords.keys()]
    coords1, coords2 = chain_coords[ch1], chain_coords[ch2]
    plddt1, plddt2 = chain_plddt[ch1], chain_plddt[ch2]

    # Calculate 2-norm
    mat = np.append(coords1, coords2, axis=0)
    a_min_b = mat[:, np.newaxis,:] - mat[np.newaxis,:,:]
    dists = np.sqrt(np.sum(a_min_b.T ** 2, axis =0)).T
    l1 = len(coords1)
    contact_dists = dists[:l1,l1:] # Upper triangular -> first dim = chain 1
    contacts = np.argwhere(contact_dists<=t)

    if contacts.shape[0] < 1:
        pdockq = 0
        ppv = 0
    else:
        #Get the average interface plDDT
        avg_if_plddt = np.average(np.concatenate([plddt1[np.unique(contacts[:,0])], plddt2[np.unique(contacts[:,1])]]))
            
        #Get the number of interface contacts
        n_if_contacts = contacts.shape[0]
        x = avg_if_plddt*np.log10(n_if_contacts)
        pdockq = 0.724 / (1 + np.exp(-0.052*(x-152.611)))+0.018

        #PPV
        PPV = np.array([0.98128027, 0.96322524, 0.95333044, 0.9400192 ,
            0.93172991, 0.92420274, 0.91629946, 0.90952562, 0.90043139,
            0.8919553 , 0.88570037, 0.87822061, 0.87116417, 0.86040801,
            0.85453785, 0.84294946, 0.83367787, 0.82238224, 0.81190228,
            0.80223507, 0.78549007, 0.77766077, 0.75941223, 0.74006263,
            0.73044282, 0.71391784, 0.70615739, 0.68635536, 0.66728511,
            0.63555449, 0.55890174])

        pdockq_thresholds = np.array([0.67333079, 0.65666073, 0.63254566, 0.62604391,
            0.60150931, 0.58313803, 0.5647381 , 0.54122438, 0.52314392,
            0.49659878, 0.4774676 , 0.44661346, 0.42628389, 0.39990988,
            0.38479715, 0.3649393 , 0.34526004, 0.3262589 , 0.31475668,
            0.29750023, 0.26673725, 0.24561247, 0.21882689, 0.19651314,
            0.17606258, 0.15398168, 0.13927677, 0.12024131, 0.09996019,
            0.06968505, 0.02946438])
        inds = np.argwhere(pdockq_thresholds>=pdockq)
            
        if len(inds)>0:
            ppv = PPV[inds[-1]][0]
        else:
            ppv = PPV[0]
    
    return pdockq, ppv




# Function to compute pDockQ2
def compute_pdockq2(plddt, pae):
    L = 1.31
    x0 = 84.733
    k = 0.075
    b = 0.005
    pDockQ2 = L / (1 + np.exp(-k * (plddt - pae / 10 - x0))) + b
    return pDockQ2


# Function to compute LIS score based on the reference code
def compute_lis(pae_matrix, chain_a_len, pae_cutoff=12.0):
    try:
        thresholded_pae = np.where(pae_matrix < pae_cutoff, 1, 0)
        scaled_pae = reverse_and_scale_matrix(pae_matrix, pae_cutoff)

        selected_values_interaction1_score = scaled_pae[:chain_a_len, chain_a_len:][thresholded_pae[:chain_a_len, chain_a_len:] == 1]
        average_selected_interaction1_score = np.mean(selected_values_interaction1_score) if selected_values_interaction1_score.size > 0 else 0

        selected_values_interaction2_score = scaled_pae[chain_a_len:, :chain_a_len][thresholded_pae[chain_a_len:, :chain_a_len] == 1]
        average_selected_interaction2_score = np.mean(selected_values_interaction2_score) if selected_values_interaction2_score.size > 0 else 0

        average_selected_interaction_total_score = (average_selected_interaction1_score + average_selected_interaction2_score) / 2

        lis = round(average_selected_interaction_total_score, 3)

        return lis
    except Exception as e:
        return None


def process_zip(zip_folder, output_dir, metrics_df, struct_pos_df, struct_neg_df, nonstruct_pos_df, nonstruct_neg_df):
    
    # Load AFM data
    zip_files = [f for f in os.listdir(zip_folder) if f.endswith('.zip')]
    for zip_file in zip_files:
        zip_path = os.path.join(zip_folder, zip_file)
        zip_name = zip_file.replace('.zip', '')
        protein_a, protein_b = zip_name.split('_')[1:3]

        # category = categorize_predictions(protein_a, protein_b, struct_pos_df, struct_neg_df, nonstruct_pos_df, nonstruct_neg_df)
        '''
        if category == -1:
            log_unmatched_proteins(protein_a, protein_b)
        '''
        nonstr_label, str_label = categorize_predictions(protein_a, protein_b)
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(output_dir)

        # Paths to extracted files
        cif_file = os.path.join(output_dir, f"{zip_name}_model_0.cif")
        full_data_json = os.path.join(output_dir, f"{zip_name}_full_data_0.json")
        summary_json = os.path.join(output_dir, f"{zip_name}_summary_confidences_0.json")

        if not (os.path.exists(cif_file) and os.path.exists(summary_json) and os.path.exists(full_data_json)):
            print(f"Files missing for {zip_name}")
            continue

        # Extract pLDDT and PAE
        with open(full_data_json, 'r') as f:
            full_data = json.load(f)

        plddt = np.mean(np.array(full_data['atom_plddts']))
        pae_matrix = np.array(full_data['pae'])
        mean_pae = np.mean(pae_matrix)
        
        print(f"PAE Matrix for {zip_name}: {pae_matrix}")
        print(f"Mean PAE for {zip_name}: {mean_pae}")
        
        # Calculate LIS
        chain_a_len = len(pae_matrix) // 2
        lis_score = compute_lis(pae_matrix, chain_a_len)

        # Read CIF file and calculate pDockQ
        chain_coords, chain_plddt = read_cif(cif_file)
        pDockQ, _ = calc_pdockq(chain_coords, chain_plddt, 8)
        print(pDockQ)
        pDockQ2 = compute_pdockq2(plddt, mean_pae)

        # Load summary for additional metrics
        with open(summary_json, 'r') as f:
            summary_data = json.load(f)
        iptm = summary_data.get('iptm', None)
        ptm = summary_data.get('ptm', None)
        ranking_confidence = summary_data.get('ranking_score', None)
        
        # Add data to DataFrame if all required metrics are available
        if None not in (pDockQ, pDockQ2, lis_score, iptm, ptm, ranking_confidence):
            metrics_df = metrics_df.append({
                'prot1': protein_a,
                'prot2': protein_b,
                'nonstr_label': nonstr_label,
                'str_label': str_label,
                'pLDDT': round(plddt, 3),
                'pAE': round(mean_pae, 3),
                'pDockQ': pDockQ,
                'pDockQ2': pDockQ2,
                'LIS': lis_score,
                'ipTM': round(float(iptm), 3),
                'pTM': round(float(ptm), 3),
                'ranking_score': round(float(ranking_confidence), 2)
            }, ignore_index=True)
    

    return metrics_df


def plot_feature_pr_curves(metrics_df, dataset_type, features):

    # Balance the number of positive and negative samples
    metrics = balance_dataset(metrics_df)


    y_true = metrics_df['label'].astype(int)

    # Inverse the interpretation of PAE 
    metrics_df['pAE'] = 1 - (metrics_df['pAE'] / 36.0)



    plt.figure(figsize=(8, 6), dpi=80)

    for feature in features:
        
        y_scores = metrics_df[feature]        

        # Compute AUCPR
        precision, recall, _ = precision_recall_curve(y_true, y_scores)
        pr_auc = auc(recall, precision)

        # Plot PR curve for the current feature
        plt.plot(recall, precision, label=f'{feature} (AUC={pr_auc:.2f})')

    # Plot formatting
    plt.title(f'AF3 PR curve ({dataset_type})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.legend(loc='best')
    plt.grid(True)
    plt.savefig(f'AF3_easy_PR_Curve_{dataset_type}.png')
    plt.close()



def plot_feature_roc_curves(metrics_df, dataset_type, features):
    
    y_true = metrics_df['label'].astype(int)

    # Inverse the interpretation of PAE
    metrics_df['pAE'] = 1 - (metrics_df['pAE'] / 36.0)


    plt.figure(figsize=(8, 6), dpi=80)

    for feature in features:

        
        y_scores = metrics_df[feature]


        # Compute ROC-AUC
        fpr, tpr, _ = roc_curve(y_true, y_scores)
        roc_auc = auc(fpr, tpr)

        # Plot ROC curve for the current feature
        plt.plot(fpr, tpr, label=f'{feature} (AUC={roc_auc:.2f})')

    # Plot formatting
    plt.plot([0, 1], [0, 1], linestyle='--', color='black', label='Random Chance')
    plt.title(f'AF3 ROC curve ({dataset_type})')
    plt.xlabel('FPR', fontsize=14)
    plt.ylabel('TPR', fontsize=14)
    plt.legend(loc='best')
    plt.grid(True)
    plt.savefig(f'AF3_easy_ROC_Curve_{dataset_type}.png')
    plt.close()

def get_optimal_threshold(y_true, y_scores):
    """
    Helper method to calculate the optimal threshold for binary classification using F1-score maximization.
    
    Parameters:
        y_true (array-like): True binary labels.
        y_scores (array-like): Predicted probability scores.

    Returns:
        float: Optimal threshold value.
    """
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_scores)
    f1_scores = (2 * precisions * recalls) / (precisions + recalls)
    optimal_idx = f1_scores.argmax()
    optimal_threshold = thresholds[optimal_idx]
    return optimal_threshold



def evaluate_promising_features(df, dataset_type):
    """
    Evaluate specified feature sets by calculating F1-score, MCC, Precision, Recall, AUPR, and AUROC.
    Uses linear regression for combining feature scores in multi-feature sets.

    Parameters:
        df (pd.DataFrame): The input dataset containing features and target column.
        dataset_type (str): Type of dataset for labeling the plot.

    Returns:
        None: Displays a bar plot comparing the performance metrics for each feature set.
    """
    # Initialize a dictionary to store metrics for each feature set
    metrics_results = {
        "Metric": ["AUPR", "AUROC", "F1-score", "MCC", "Precision", "Recall"],
        "ipTM": [],
        "pDockQ": [],
        "ipTM + pDockQ": [],
        "All 8 Features": []
    }

    feature_sets = {
        "ipTM": ['ipTM'],
        "pDockQ": ['pDockQ'],
        "ipTM + pDockQ": ['ipTM', 'pDockQ'],
        "All 8 Features": ['pLDDT', 'pAE', 'LIS', 'pDockQ', 'pDockQ2', 'ipTM', 'pTM', 'ranking_score']
    }
    
    coefficients = {}

    # Define the target variable
    y_true = df['label']

    # Evaluate each feature set
    for set_name, feature_set in feature_sets.items():
        X_set = df[feature_set]

        # If multiple features, use linear regression to combine them into a single score
        if X_set.shape[1] > 1:
            # Fit linear regression to create a weighted combination of features
            lr = LinearRegression()
            lr.fit(X_set, y_true)
            y_scores = lr.predict(X_set)
        
            coefficients[set_name] = dict(zip(feature_set, lr.coef_))

        else:
            # Use single feature directly as scores
            y_scores = X_set.squeeze()

        # Calculate AUPR and AUROC directly on the continuous scores
        metrics_results[set_name].append(average_precision_score(y_true, y_scores))
        metrics_results[set_name].append(roc_auc_score(y_true, y_scores))

        # Convert continuous scores into binary predictions using a probability threshold for calculating F1, MCC, Precision, and Recall
        optimal_threshold = get_optimal_threshold(y_true, y_scores)
        y_pred = (y_scores >= optimal_threshold).astype(int)

        # Calculate other metrics
        metrics_results[set_name].append(f1_score(y_true, y_pred))
        metrics_results[set_name].append(matthews_corrcoef(y_true, y_pred))
        metrics_results[set_name].append(precision_score(y_true, y_pred))
        metrics_results[set_name].append(recall_score(y_true, y_pred))

    # Display the coefficients for multi-feature sets
    for set_name, coef_dict in coefficients.items():
        for feature, coef in coef_dict.items():
            print(f"{set_name} - {feature}: {coef:.4f}")

    # Convert metrics results to a DataFrame for easier plotting
    metrics_df = pd.DataFrame(metrics_results)

    # Define custom colors for each feature set to match the style shown
    colors = {
        "ipTM": "#FFB300",  # Orange
        "pDockQ": "#FF8000",  # Dark Orange
        "ipTM + pDockQ": "#FF4080",  # Pink
        "All 8 Features": "#00A0E0"  # Light Blue
    }

    # Plotting the metrics in a transposed histogram
    fig, ax = plt.subplots(figsize=(12, 8))
    metrics_df.set_index("Metric").plot(kind="bar", ax=ax, color=[colors[fs] for fs in feature_sets.keys()])
    ax.set_title(f"Performance Evaluation for AF3 Features ({dataset_type})")
    ax.set_xlabel("Metric")
    ax.set_ylabel("Score")
    plt.xticks(rotation=45)
    plt.legend(title="Feature Set", loc="best")
    plt.tight_layout()
    plt.savefig(f'AF3_easy_{dataset_type}_ipTM+pDockQ_evaluation.png')



# Function to plot histograms for the specified features and save them to .png
def plot_histograms(df, dataset_type):
    
    # Metrics to plot (features)
    features = ['pLDDT', 'pAE', 'pDockQ', 'pDockQ2', 'LIS', 'ipTM', 'pTM', 'ranking_score']

    # Define colors based on dataset type
    pos_color_key = 'struct_pos' if dataset_type == 'structural' else 'nonstruct_pos'
    neg_color_key = 'struct_neg' if dataset_type == 'structural' else 'nonstruct_neg'


    # Loop through structural and nonstructural datasets
    for feature in features:
            
        plt.figure(figsize=(12, 6))

        # Separate positive and negative samples based on label
        df_pos = df[df['label'] == 1]
        df_neg = df[df['label'] == 0]

        # Debugging: print min, max, and unique values for each feature
        print(f"{feature} for {dataset_type}:")
        print("Positive min/max:", df_pos[feature].min(), df_pos[feature].max())
        print("Negative min/max:", df_neg[feature].min(), df_neg[feature].max())
        print("Unique values in Positive:", len(df_pos[feature].unique()))
        print("Unique values in Negative:", len(df_neg[feature].unique()))

        # Manually set bins to ensure the entire range is covered
        bins = np.linspace(min(df_pos[feature].min(), df_neg[feature].min()),
                               max(df_pos[feature].max(), df_neg[feature].max()), 30)
        width = (bins[1] - bins[0]) / 3  # Narrower bars for better side-by-side view

        # Use the colors provided in the global dictionary
        pos_color = colors[pos_color_key]
        neg_color = colors[neg_color_key]
        
         # Calculate percentage histograms for positive and negative samples
        pos_counts, _ = np.histogram(df_pos[feature], bins=bins)
        neg_counts, _ = np.histogram(df_neg[feature], bins=bins)

        # Convert counts to percentages
        pos_percentages = (pos_counts / len(df_pos)) * 100  # Percentage for positive samples
        neg_percentages = (neg_counts / len(df_neg)) * 100  # Percentage for negative samples


        # Plot histograms for positive and negative samples
        plt.bar(bins[:-1], pos_percentages, width=width, alpha=0.6, color=pos_color, label=f"{dataset_type} Positive")
        plt.bar(bins[:-1] + width, neg_percentages, width=width, alpha=0.6, color=neg_color, label=f"{dataset_type} Negative")

        plt.title(f"AF3 {feature} distribution ({dataset_type})", fontsize=20)
        plt.xlabel(f"{feature}", fontsize=18)
        plt.ylabel("Density (%)", fontsize=18)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.legend(loc="best", fontsize=16)
        plt.grid(True)

        # Save the plot
        plt.savefig(f"AF3_easy_{dataset_type}_{feature}_histogram.png")
        plt.close()

# Function to assign labels based on existing structural and nonstructural criteria
def assign_labels(df):
    """
    Assign labels for combined structural and nonstructural data.
    If either str_label or nonstr_label is 1, assign 1 as the label.
    If both are 0, assign 0 as the label.
    """
    df['label'] = df.apply(lambda row: 1 if (row['str_label'] == 1 or row['nonstr_label'] == 1) else 0, axis=1)
    return df

def create_train_test_split(df, label_column, features):
    """
    Create train-test splits within a given structural or nonstructural dataset using predefined features.

    Args:
        df (pd.DataFrame): The balanced dataset.
        label_column (str): The column used for the labels (e.g., str_label or nonstr_label).
        features (list): The list of feature columns to be used for training.

    Returns:
    # Use predefined feature columns from `columns_to_use` in the main function
    """

    # Filter the DataFrame to include only the specified feature columns
    X = df[features]  # Only use columns from `columns_to_use`
    
    # Ensure labels are extracted as 1D array
    y = df[label_column].values.flatten()        
     
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("After splitting (currnely in create_train_test_split()):")
    print(f"X_train shape: {X_train.shape}")  # Should match the number of samples and features (e.g., (1008, 8))
    print(f"y_train shape: {y_train.shape}")  # Should be 1D (e.g., (1008,))
    print(f"X_test shape: {X_test.shape}")
    print(f"y_test shape: {y_test.shape}")
    


    return X_train, X_test, y_train, y_test


def plot_correlation_matrix(df, features, dataset_type, method):
    X = df[features]

    # Calculate correlation matrix
    correlation_matrix = X.corr()

    # Visualize the correlation matrix using clustermap for hierarchical clustering + correlation matrix
    sns.clustermap(correlation_matrix, method="complete", cmap='RdBu', annot=True, vmin=-1, vmax=1, annot_kws={"size": 7}, figsize=(15,12))
    plt.savefig(f'AF3_easy_{dataset_type}_hierarchical_clustering.png')
    # Step 2: Visualize the correlation matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
    plt.title(f'{method.capitalize()} correlation matrix ({dataset_type})')
    plt.savefig(f'AF3_easy_{dataset_type}_correlation_matrix.png')

    distance_matrix = 1 - np.abs(correlation_matrix)

    # Perform hierarchical clustering using the average linking method
    linkage_matrix = linkage(distance_matrix, method='ward', metric='euclidean')
    # Visualize the hierarchical clustering using a dendrogram
    plt.figure(figsize=(12,8))
    dendrogram(linkage_matrix, labels=features, leaf_rotation=90)
    plt.title(f"Hierarchical clustering dendrogram of features ({dataset_type})")
    plt.xlabel('Features')
    plt.ylabel('Distance')
    plt.savefig(f'AF3_easy_{dataset_type}_hierarchical_clusterings.png')

# STEP 1: Calculate correlation matrices
def calculate_correlation_matrix(X, method):
    """Calculate correlation matrix based on specified method: pearson, spearman, or kendall."""
    print(f"Calculating {method} correlation matrix.")
    
    if method == "pearson":
        corr_matrix = X.corr(method='pearson')
    elif method == "spearman":
        corr_matrix = X.corr(method='spearman')
    elif method == "kendall":
        corr_matrix = X.corr(method='kendall')
    
    # Debugging: Print correlation matrix
    print(f"Correlation Matrix ({method}):")
    print(corr_matrix)
    
    return corr_matrix

# STEP 3: Perform cross-validation and compute AUC on each threshold
def compute_optimal_threshold(X, y, corr_matrix, threshold):

    corr_matrix = np.abs(corr_matrix)

    # Step 1: Filter features based on correlation with the label
    label_correlations = np.abs(X.apply(lambda col: col.corr(y)))
    
    # Step 2: Select features based on their correlation with the label
    selected_features = X.columns[label_correlations > threshold]

    if len(selected_features) == 0:
        print(f"No features selected at threshold {threshold}")
        return 0, [], {}

    X_selected = X[selected_features]


    # Step 3: Perform 5-fold cross validation using RandomForest and AUC
    clf = RandomForestClassifier(random_state=42)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    auc_scores = cross_val_score(clf, X_selected, y, cv=cv, scoring='roc_auc')
    mean_auc = np.mean(auc_scores)

    return mean_auc, selected_features, {}



# Function to plot using seaborn and dictionaries
def plot_auc_threshold(corr_matrix, matrix_name, X, y, features, dataset_type):
    thresholds = np.arange(0.0, 0.51, 0.1)  # Adjusted to step by 0.1 for clearer plot
    auc_scores = []
    selected_features_dict = {}

    # Loop through thresholds and store selected features in the dictionary
    for threshold in thresholds:
        auc, selected_features, _ = compute_optimal_threshold(X, y, np.abs(corr_matrix), threshold)
        if auc is not None:
            auc_scores.append(auc)
            selected_features_dict[threshold] = list(selected_features)  # Save selected features in dictionary

    # Prepare data for seaborn plotting
    thresholds_list, num_selected_features = [], [] 
    
    feature_presence = {feature: [] for feature in X.columns}  # Initialize empty lists for each feature

    # Counting the number of selected features at each threshold and tracking individual feature selection
    for threshold in thresholds:
        thresholds_list.append(threshold)
        selected_features = selected_features_dict.get(threshold, [])
        num_selected_features.append(len(selected_features))
        # Record whether each feature is selected (1) or not (0) at this threshold
        for feature in X.columns:
            feature_presence[feature].append(1 if feature in selected_features else 0)

    plot_data = pd.DataFrame({
        'Thresholds': thresholds_list,
        'Num_Features': num_selected_features,
        'AUC': auc_scores
    })

    # Convert feature presence into a DataFrame for stacked bar plotting
    feature_presence_df = pd.DataFrame(feature_presence)
    feature_presence_df['Thresholds'] = thresholds_list

    # Create the figure
    fig, ax1 = plt.subplots(figsize=(10, 6))
    width = 0.07 

    # Plot stacked bars for selected features at each threshold
    feature_colors = {
        "LIS": "#1f77b4", "ipTM": "#ff7f0e", "pTM": "#2ca02c", "pLDDT": "#d62728",
        "pDockQ": "#9467bd", "pDockQ2": "#8c564b", "pAE": "#e377c2", "ranking_score": "#7f7f7f"
    }

    bottom = np.zeros(len(thresholds))  # Initialize the bottom for stacking bars

    # Plot each feature as a portion of the stacked bar
    for feature, color in feature_colors.items():
        ax1.bar(thresholds_list, feature_presence_df[feature], width=width, bottom=bottom, color=color, label=feature)
        bottom += feature_presence_df[feature]  # Stack the features vertically

    ax1.set_ylabel('# Selected Features', fontsize=14)
    ax1.set_xlabel('Thresholds', fontsize=14)

    # Create the line plot for AUC on secondary axis
    ax2 = ax1.twinx()
    sns.lineplot(x='Thresholds', y='AUC', data=plot_data, marker='o', color='blue', ax=ax2)
    ax2.set_ylabel('AUC', fontsize=14)

    # Customize the plot
    plt.title(f'AF3 AUC vs thresholds Feature selection ({dataset_type})', fontsize=16)
    plt.xticks(rotation=60)

    # Add legend
    ax1.legend(loc='upper right', title="Selected Features")

    plt.tight_layout()
    plt.savefig(f"AF3 easy {matrix_name} AUC vs Threshold({dataset_type})")
    return selected_features_dict


# Define function to perform feature selection based on correlation threshold
def correlation_based_feature_selection(df, corr_matrix, threshold):
    """Select features based on correlation threshold."""
    selected_features = []
    for i, row in enumerate(corr_matrix):
        if any(row > threshold):
            selected_features.append(i)
    return df.iloc[:, selected_features]

# Apply feature selection based on correlation method and threshold
def apply_feature_selection(df, method, threshold):
    """Apply correlation-based feature selection with specified method and threshold."""
    correlation_matrix = calculate_correlation_matrix(df, method)
    selected_df = correlation_based_feature_selection(df, correlation_matrix, threshold)
    selected_features = selected_df.columns
    print(f"Selected {len(selected_features)} features for {method} at threshold {threshold:.2f}")
    return selected_features

# Function to evaluate models using GridSearchCV
def evaluate_models_with_gridcv(X_train, X_test, y_train, y_test):
    """Perform hyperparameter tuning and evaluate models using GridSearchCV."""
    param_grids = {
        'Random Forest': {
            'n_estimators': [10, 50],
            'max_depth': [None, 10],
            'min_samples_split': [2, 5],
            'min_samples_leaf': [1, 2],
            'bootstrap': [True],
            'max_features': ['sqrt']
        },
        'SVM': {
            'C': [0.1, 1],
            'gamma': ['scale'],
            'kernel': ['rbf']
        },
        'MLP': {
            'hidden_layer_sizes': [(100,)],
            'activation': ['relu'],
            'solver': ['adam'],
            'alpha': [0.0001],
            'learning_rate': ['constant'],
            'max_iter': [200]
        }
    }
    
    classifiers = {
        'Random Forest': RandomForestClassifier(random_state=42),
        'SVM': SVC(probability=True, random_state=42),
        'MLP': MLPClassifier(max_iter=1000, random_state=42)
    }

    results = {}
    for name, clf in classifiers.items():
        grid_search = GridSearchCV(estimator=clf, param_grid=param_grids[name],
                                   cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
                                   scoring='roc_auc', n_jobs=-1, verbose=1)
        grid_search.fit(X_train, y_train)

        best_clf = grid_search.best_estimator_
        y_pred = best_clf.predict(X_test)
        y_probs = best_clf.predict_proba(X_test)[:, 1]

        roc_auc = roc_auc_score(y_test, y_probs)
        precision, recall, _ = precision_recall_curve(y_test, y_probs)
        pr_auc = auc(recall, precision)
        f1 = f1_score(y_test, y_pred)
        mcc = matthews_corrcoef(y_test, y_pred)
        accuracy = accuracy_score(y_test, y_pred)

        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        tpr = tp / (tp + fn)
        tnr = tn / (tn + fp)

        clf_results = {
            'ROC AUC': roc_auc,
            'PR AUC': pr_auc,
            'F1 Score': f1,
            'MCC': mcc,
            'Accuracy': accuracy,
            'TPR (Recall)': tpr,
            'TNR (Specificity)': tnr
        }

        results[name] = clf_results

    return results

# Function to visualize results in a table format
def visualize_results_table(results_list, dataset_type):
    """Visualize the results in a table format similar to Table 2."""
    table_data = []
    for method, threshold, num_features, results in results_list:
        for model_name, res in results.items():
            if method == 'None':
                feature_selection_desc = "No Feature Selection"
            else:
                feature_selection_desc = f"{method.upper()} (threshold = {threshold:.2f})"
            
            row = {
                'Model': model_name,
                'Feature Selection': feature_selection_desc,
                'Feature #': num_features,
                'TPR': round(res['TPR (Recall)'], 3),
                'TNR': round(res['TNR (Specificity)'], 3),
                'ACC': round(res['Accuracy'], 3),
                'F1': round(res['F1 Score'], 3),
                'MCC': round(res['MCC'], 3),
                'AUC': round(res['ROC AUC'], 3),
                'AUPR': round(res['PR AUC'], 3)
            }
            table_data.append(row)

    table_df = pd.DataFrame(table_data)
    table_df = table_df.sort_values(by=['Model', 'Feature Selection'])

    print(table_df.to_string(index=False))
    table_df.to_csv("AF3_easy_results_table.csv", index=False)
    print("\nResults have been saved to 'AF3_results_table.csv'.")



def normalize_scores(df):
    """
    We aim to normalize pLDDT, ranking_score, and pAE
    """

    norm_scores = ['pLDDT', 'ranking_score', 'pAE']

    df['pAE'] = 1 - (df['pAE'] / 36.0)
    df['pLDDT'] = df['pLDDT'] / 100.0
    df['ranking_score'] = (df['ranking_score'] + 100.0) / (1.5 + 100.0) 

    return df

                        
def main():
    print('--------------------------- STARTING AF3 ANALYSIS ---------------------------')

    # 1. Load AFM scores 
    af3_df = pd.read_csv('/home/jc3668/AF3_scores.csv')
    output_dir = '/home/jc3668/projects/af3_analysis'
    
    # 2. Normalize a few AF features(pLDDT, pAE, and ranking_score)
    af3_df = normalize_scores(af3_df)

    print("Checking for invalid labels in the dataset...")
    print(f"total # samples: {len(af3_df)}")
    print(f"Structural Positives: {len(af3_df[af3_df['str_label'] == 1])}")
    print(f"Structural Negatives: {len(af3_df[af3_df['str_label'] == 0])}")
    print(f"Nonstructural Positives: {len(af3_df[af3_df['nonstr_label'] == 1])}")
    print(f"Nonstructural Negatives: {len(af3_df[af3_df['nonstr_label'] == 0])}")

    columns_to_use = ['LIS', 'ipTM', 'pTM', 'pLDDT', 'pDockQ', 'pAE', 'pDockQ2', 'ranking_score']

    # Create balanced structural and nonstructural datasets
    struct_df = af3_df[(af3_df['str_label'] == 1) | (af3_df['str_label'] == 0)]
    nonstruct_df = af3_df[(af3_df['nonstr_label'] == 1) | (af3_df['nonstr_label'] == 0)]

    struct_df['label'] = struct_df['str_label']  
    nonstruct_df['label'] = nonstruct_df['nonstr_label']  


    # Plot histogram
    plot_histograms(struct_df, "structural")
    plot_histograms(nonstruct_df, "nonstructural")


    # Draw ROC/PR-AUC analyses. Make sure to balance dataset using bootstrapping for PR-curve. 
    plot_feature_roc_curves(struct_df, "structural", columns_to_use)
    plot_feature_roc_curves(nonstruct_df, "nonstructural", columns_to_use)

    plot_feature_pr_curves(struct_df, "structural", columns_to_use)
    plot_feature_pr_curves(nonstruct_df, "nonstructural", columns_to_use)

    

    # Plot ipTM & ranking_score 
    evaluate_promising_features(struct_df, "structural")
    evaluate_promising_features(nonstruct_df, "nonstructural")
        
    # For machine learning tasks, create train-test splits *within* balanced structural and nonstructural subsets
    X_struct = struct_df[columns_to_use]
    y_struct = struct_df['label']

    X_struct_train, X_struct_test, y_struct_train, y_struct_test = train_test_split(X_struct, y_struct, test_size=0.2, random_state=42)
    
    X_nonstruct = nonstruct_df[columns_to_use]
    y_nonstruct = nonstruct_df['label']

    X_nonstr_train, X_nonstr_test, y_nonstr_train, y_nonstr_test = train_test_split(X_nonstruct, y_nonstruct, test_size = 0.2, random_state = 42)


    print(f"After splitting - X_struct_train shape: {X_struct_train.shape}") #After splitting - X_struct_train shape: (1008, 8)
    print(f"After splitting - y_struct_train shape: {y_struct_train.shape}")  #After splitting - y_struct_train shape: (1008, 8)

    print(f"After splitting - X_nonstruct_train shape: {X_nonstr_train.shape}") #After splitting - X_struct_train shape: (1008, 8)
    print(f"After splitting - y_nonstruct_train shape: {y_nonstr_train.shape}")  #After splitting - y_struct_train shape: (1008, 8)


    correlation_methods = ['pearson', 'spearman', 'kendall']
    
    for method in correlation_methods:
        # STEP 1: Calculate correlation matrix
        corr_matrix = calculate_correlation_matrix(X_struct_train, method)

        # STEP 2: Plot correlation matrix
        plot_correlation_matrix(struct_df, columns_to_use, "structural", method)
        plot_correlation_matrix(nonstruct_df, columns_to_use, "nonstructrual", method)
        
        # STEP 4: Plot AUC vs Threshold and find the best threshold
        plot_auc_threshold(corr_matrix, method, X_struct_train, y_struct_train, columns_to_use, "structural")


   # WE do the same for nonstructural dataset.                         
##############################################################################################################################



    print('-----------------------------------------------------------')
    #X_nonstr_train, X_nonstr_test, y_nonstr_train, y_nonstr_test = create_train_test_split(

    #   nonstruct_df, 'label', columns_to_use
    #)
    
    
    #print(f"After splitting - X_nonstr_train shape: {X_nonstr_train.shape}")
    #print(f"After splitting - y_nonstr_train shape: {y_nonstr_train.shape}")

    

    print('--------------------------- AF3 ANALYSIS COMPLETE ---------------------------')

if __name__ == "__main__":
    main()

