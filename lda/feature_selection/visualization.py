"""
Centralized visualization functions for all feature selection algorithms.
Provides consistent diagnostic plots across Fisher_AMINO, Chi_sq_AMINO, BPSO, and MPSO.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

def visualize_feature_selection_diagnostics(scores, candidate_df, final_features, target_col, algorithm_name="AMINO"):
    """
    Unified diagnostic dashboard for ALL feature selection algorithms.
    
    Parameters:
    -----------
    scores : pd.Series
        Feature scores (Fisher or Chi-squared)
    candidate_df : pd.DataFrame
        DataFrame with candidate features
    final_features : list
        List of final selected features
    target_col : str
        Target column name
    algorithm_name : str
        Name of algorithm for labeling ("AMINO", "BPSO", "MPSO")
    """
    print(f"📊 Generating {algorithm_name} diagnostic plots...")
    try:
        sns.set_theme(style="whitegrid")
        fig, axes = plt.subplots(1, 3, figsize=(21, 6))
        
        # 1. Signal Strength (Scree Plot)
        sorted_scores = scores.sort_values(ascending=False)
        axes[0].plot(range(len(sorted_scores)), sorted_scores.values, color='#333333', lw=1, alpha=0.5)
        axes[0].fill_between(range(len(sorted_scores)), sorted_scores.values, color='#333333', alpha=0.1)
        
        # Highlight final selected features
        if len(final_features) > 0:
            selected_indices = [sorted_scores.index.get_loc(f) for f in final_features if f in sorted_scores.index]
            axes[0].scatter(selected_indices, sorted_scores.iloc[selected_indices], color='red', s=45, label=f'{algorithm_name} Selected', zorder=5)
            axes[0].legend()
            
        score_type = "Fisher" if "Fisher" in str(type(scores.iloc[0])) else "Chi-Sq"
        axes[0].set_title(f"Feature Signal Strength ({score_type})", fontsize=14)
        axes[0].set_xlabel("Feature Rank")
        axes[0].set_ylabel(f"{score_type} Score")
        axes[0].set_yscale('log')
        
        # 2. Redundancy (Correlation Heatmap)
        if len(final_features) > 1:
            corr = candidate_df[final_features].corr()
            sns.heatmap(corr, cmap="coolwarm", center=0, ax=axes[1], annot=False)
            axes[1].set_title("Feature Redundancy (Correlation)", fontsize=14)
        else:
            axes[1].text(0.5, 0.5, "Need 2+ Features\nfor Heatmap", ha='center')

        # 3. State Space Mapping (2D Scatter)
        if len(final_features) >= 2:
            sample_df = candidate_df.sample(min(2000, len(candidate_df)))
            sns.scatterplot(data=sample_df, x=final_features[0], y=final_features[1], hue=target_col, palette="deep", s=20, alpha=0.7, ax=axes[2])
            axes[2].set_title(f"State Space Mapping\n{final_features[0]} vs {final_features[1]}", fontsize=14)
        else:
            axes[2].text(0.5, 0.5, "Need 2+ Features\nfor Scatter Plot", ha='center')
        
        plt.tight_layout()
        plt.show()
    except Exception as e:
        print(f"Visualization failed: {e}")

# Backward compatibility wrappers
def visualize_amino_diagnostics(scores, candidate_df, final_features, target_col, score_type="Fisher"):
    """Backward compatibility wrapper for AMINO algorithms."""
    return visualize_feature_selection_diagnostics(scores, candidate_df, final_features, target_col, "AMINO")

def visualize_mpso_diagnostics(final_df, fisher_scores, candidates, final_sel, target_col):
    """Backward compatibility wrapper for MPSO."""
    # Extract selected features from MPSO selection matrix
    contributing_mask = np.any(final_sel, axis=1)
    selected_features = [candidates[i] for i, m in enumerate(contributing_mask) if m]
    return visualize_feature_selection_diagnostics(fisher_scores, final_df, selected_features, target_col, "MPSO")

def visualize_bpso_diagnostics(final_df, fisher_scores, candidates, selected_features, target_col):
    """Backward compatibility wrapper for BPSO."""
    return visualize_feature_selection_diagnostics(fisher_scores, final_df, selected_features, target_col, "BPSO")

def visualize_fisher_amino_diagnostics(fisher_series, candidate_df, final_features, target_col):
    """Backward compatibility wrapper for Fisher AMINO."""
    return visualize_feature_selection_diagnostics(fisher_series, candidate_df, final_features, target_col, "AMINO")

def visualize_chi_sq_amino_diagnostics(chi_scores, candidate_df, final_features, target_col):
    """Backward compatibility wrapper for Chi-squared AMINO."""
    return visualize_feature_selection_diagnostics(chi_scores, candidate_df, final_features, target_col, "AMINO")