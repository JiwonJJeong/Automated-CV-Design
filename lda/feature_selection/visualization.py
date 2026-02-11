"""
Centralized visualization functions for all feature selection algorithms.
Provides consistent diagnostic plots across Fisher_AMINO, Chi_sq_AMINO, BPSO, and MPSO.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

def visualize_amino_diagnostics(scores, candidate_df, final_features, target_col, score_type="Fisher"):
    """
    Standardized diagnostic dashboard for AMINO-based Feature Selection results.
    
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
    score_type : str
        Type of score ("Fisher" or "Chi-Sq")
    """
    print("📊 Generating AMINO diagnostic plots...")
    try:
        sns.set_theme(style="whitegrid")
        fig, axes = plt.subplots(1, 3, figsize=(21, 6))
        
        # 1. Signal Strength (Scree Plot)
        sorted_scores = scores.sort_values(ascending=False)
        axes[0].plot(range(len(sorted_scores)), sorted_scores.values, color='#333333', lw=1, alpha=0.5)
        axes[0].fill_between(range(len(sorted_scores)), sorted_scores.values, color='#333333', alpha=0.1)
        
        # Highlight final AMINO selected features
        if len(final_features) > 0:
            selected_indices = [sorted_scores.index.get_loc(f) for f in final_features if f in sorted_scores.index]
            axes[0].scatter(selected_indices, sorted_scores.iloc[selected_indices], color='red', s=45, label='AMINO Selected', zorder=5)
            axes[0].legend()
            
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
            f1, f2 = final_features[0], final_features[1]
            sample_df = candidate_df.sample(min(2000, len(candidate_df)))
            sns.scatterplot(data=sample_df, x=f1, y=f2, hue=target_col, palette="deep", s=20, alpha=0.7, ax=axes[2])
            axes[2].set_title(f"State Space Mapping\n{f1} vs {f2}", fontsize=14)
        else:
            axes[2].text(0.5, 0.5, "Need 2+ Features\nfor Scatter Plot", ha='center')
        
        plt.tight_layout()
        plt.show()
    except Exception as e:
        print(f"Visualization failed: {e}")

def visualize_mpso_diagnostics(final_df, fisher_scores, candidates, final_sel, target_col):
    """
    Separate visualization function for MPSO diagnostics.
    
    Parameters:
    -----------
    final_df : pd.DataFrame
        Final DataFrame with MPSO projected dimensions
    fisher_scores : pd.Series
        Fisher scores for all features
    candidates : list
        List of candidate features
    final_sel : np.ndarray
        Binary selection matrix from MPSO
    target_col : str
        Target column name
    """
    print("📊 Generating MPSO diagnostic plots...")
    try:
        fig, axes = plt.subplots(1, 3, figsize=(21, 6))
        sns.set_theme(style="whitegrid")
        
        # 1. Signal Strength (Fisher Scree Plot)
        y_vals = fisher_scores.values
        axes[0].plot(range(len(y_vals)), y_vals, color='grey', alpha=0.5)
        candidate_indices = [fisher_scores.index.get_loc(c) for c in candidates if c in fisher_scores.index]
        axes[0].scatter(candidate_indices, fisher_scores.iloc[candidate_indices], color='orange', s=20, alpha=0.6, label='Candidates')
        
        # Highlight top MPSO contributors
        top_contributor_mask = np.any(final_sel, axis=1)
        top_contributor_indices = [fisher_scores.index.get_loc(candidates[i]) for i, m in enumerate(top_contributor_mask) if m]
        axes[0].scatter(top_contributor_indices, fisher_scores.iloc[top_contributor_indices], color='red', s=45, label='Top Contributors', zorder=5)
        
        axes[0].set_title("Feature Signal Strength (Fisher)", fontsize=14)
        axes[0].set_xlabel("Feature Rank")
        axes[0].set_ylabel("Fisher Score")
        axes[0].legend()
        
        # 2. Redundancy (Correlation of Projected Dimensions)
        proj_cols = [c for c in final_df.columns if c.startswith('MPSO_')]
        if len(proj_cols) > 1:
            corr = final_df[proj_cols].corr()
            sns.heatmap(corr, cmap="coolwarm", center=0, ax=axes[1], annot=False)
            axes[1].set_title("Projected Dimension Redundancy", fontsize=14)
        else:
            axes[1].text(0.5, 0.5, "Need 2+ Dimensions\nfor Heatmap", ha='center')
        
        # 3. Final State Space (2D Scatter of Top 2 Dimensions)
        if len(proj_cols) >= 2:
            sample_df = final_df.sample(min(2000, len(final_df)))
            sns.scatterplot(data=sample_df, x=proj_cols[0], y=proj_cols[1], hue=target_col, palette="deep", s=20, alpha=0.7, ax=axes[2])
            axes[2].set_title(f"Final State Space\n{proj_cols[0]} vs {proj_cols[1]}", fontsize=14)
        else:
            axes[2].text(0.5, 0.5, "Need 2+ Dimensions\nfor Scatter Plot", ha='center')
        
        plt.tight_layout()
        plt.show()
    except Exception as e:
        print(f"Visualization failed: {e}")

def visualize_bpso_diagnostics(final_df, fisher_scores, candidates, selected_features, target_col):
    """
    Visualization function for BPSO diagnostics.
    
    Parameters:
    -----------
    final_df : pd.DataFrame
        Final DataFrame with selected features
    fisher_scores : pd.Series
        Fisher scores for all features
    candidates : list
        List of candidate features
    selected_features : list
        List of final selected features
    target_col : str
        Target column name
    """
    print("📊 Generating BPSO diagnostic plots...")
    try:
        fig, axes = plt.subplots(1, 3, figsize=(21, 6))
        sns.set_theme(style="whitegrid")
        
        # 1. Signal Strength (Fisher Scree Plot)
        y_vals = fisher_scores.values
        axes[0].plot(range(len(y_vals)), y_vals, color='grey', alpha=0.5)
        candidate_indices = [fisher_scores.index.get_loc(c) for c in candidates if c in fisher_scores.index]
        axes[0].scatter(candidate_indices, fisher_scores.iloc[candidate_indices], color='orange', s=20, alpha=0.6, label='Candidates')
        
        # Highlight BPSO selected features
        selected_indices = [fisher_scores.index.get_loc(f) for f in selected_features if f in fisher_scores.index]
        axes[0].scatter(selected_indices, fisher_scores.iloc[selected_indices], color='red', s=45, label='BPSO Selected', zorder=5)
        
        axes[0].set_title("Feature Signal Strength (Fisher)", fontsize=14)
        axes[0].set_xlabel("Feature Rank")
        axes[0].set_ylabel("Fisher Score")
        axes[0].legend()
        
        # 2. Redundancy (Correlation of Selected Features)
        if len(selected_features) > 1:
            corr = final_df[selected_features].corr()
            sns.heatmap(corr, cmap="coolwarm", center=0, ax=axes[1], annot=False)
            axes[1].set_title("Selected Features Redundancy", fontsize=14)
        else:
            axes[1].text(0.5, 0.5, "Need 2+ Features\nfor Heatmap", ha='center')
        
        # 3. Final State Space (2D Scatter of Top 2 Features)
        if len(selected_features) >= 2:
            sample_df = final_df.sample(min(2000, len(final_df)))
            sns.scatterplot(data=sample_df, x=selected_features[0], y=selected_features[1], hue=target_col, palette="deep", s=20, alpha=0.7, ax=axes[2])
            axes[2].set_title(f"Final State Space\n{selected_features[0]} vs {selected_features[1]}", fontsize=14)
        else:
            axes[2].text(0.5, 0.5, "Need 2+ Features\nfor Scatter Plot", ha='center')
        
        plt.tight_layout()
        plt.show()
    except Exception as e:
        print(f"Visualization failed: {e}")

# Convenience functions for backward compatibility
def visualize_fisher_amino_diagnostics(fisher_series, candidate_df, final_features, target_col):
    """Backward compatibility wrapper for Fisher AMINO."""
    return visualize_amino_diagnostics(fisher_series, candidate_df, final_features, target_col, "Fisher")

def visualize_chi_sq_amino_diagnostics(chi_scores, candidate_df, final_features, target_col):
    """Backward compatibility wrapper for Chi-squared AMINO."""
    return visualize_amino_diagnostics(chi_scores, candidate_df, final_features, target_col, "Chi-Sq")