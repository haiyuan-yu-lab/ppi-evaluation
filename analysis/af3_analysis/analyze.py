import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

colors = {
    "nonstruct_neg": "#EF4035",
    "nonstruct_pos": "#6EB43F",
    "struct_neg": "#F8981D",
    "struct_pos": "#006699"
}


def plot_histograms(df, dataset_type):

    # Metrics to plot (features)
    features = ['mean_pLDDT', 'best_pLDDT', 'mean_pAE', 'mean_interface_pLDDT', 'best_interface_pLDDT',
                'mean_interface_pAE', 'pDockQ', 'mean_pDockQ2', 'best_pDockQ2', 'mean_LIS', 'best_LIS',
                'mean_LIA', 'best_LIA', 'ipTM', 'pTM', 'ranking_score']

    # Define colors based on dataset type
    pos_color_key = 'struct_pos' if dataset_type == 'structural' else 'nonstruct_pos'
    neg_color_key = 'struct_neg' if dataset_type == 'structural' else 'nonstruct_neg'

    # Define global colors (replace with your color definitions)
    colors = {'struct_pos': 'blue', 'struct_neg': 'orange',
              'nonstruct_pos': 'green', 'nonstruct_neg': 'red'}

    # Loop through features to create histograms
    for feature in features:
        fig, ax = plt.subplots(figsize=(12, 6))

        # Separate positive and negative samples based on label
        df_pos = df[df['label'] == 1]
        df_neg = df[df['label'] == 0]

        # Debugging: print min, max, and unique values for each feature
        print(f"{feature} for {dataset_type}:")
        print("Positive min/max:", df_pos[feature].min(), df_pos[feature].max())
        print("Negative min/max:", df_neg[feature].min(), df_neg[feature].max())

        # Define bins based on the range of data, starting from 0
        data_min = max(0, min(df_pos[feature].min(), df_neg[feature].min()))  # Exclude negative values
        data_max = max(df_pos[feature].max(), df_neg[feature].max())
        bins = np.linspace(data_min, data_max, 30)
        width = (bins[1] - bins[0]) / 3  # Narrow bars for separation

        # Calculate percentages for normalized density
        pos_counts, _ = np.histogram(df_pos[feature], bins=bins)
        neg_counts, _ = np.histogram(df_neg[feature], bins=bins)
        pos_percentages = (pos_counts / len(df_pos)) * 100
        neg_percentages = (neg_counts / len(df_neg)) * 100

        # Plot histograms side-by-side for positive and negative samples
        ax.bar(bins[:-1], pos_percentages, width=width, color=colors[pos_color_key],
               alpha=0.6, label=f"{dataset_type} Positive")
        ax.bar(bins[:-1] + width, neg_percentages, width=width, color=colors[neg_color_key],
               alpha=0.6, label=f"{dataset_type} Negative")

        if '_' in feature:
            feature_name = feature.split('.')

        # Set title, labels, and legends
        ax.set_title(f"AF3 {feature_name[i] + ' '  for i in feature_name} distribution ({dataset_type})", fontsize=20)
        ax.set_xlabel(f"{feature}", fontsize=18)
        ax.set_ylabel("Density (%)", fontsize=18)
        ax.legend(loc="best", fontsize=16)
        ax.grid(axis='y')
        ax.set_facecolor('white')  # Clean background

        # Save the plot
        plt.savefig(f"figure/distributions/AF3_{dataset_type}_{feature}_histogram.png")
        print(f"Histogram for {feature} complete!")
        plt.close()

