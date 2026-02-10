import sys
import os
import numpy as np
import pandas as pd
import gc
from kneed import KneeLocator
import matplotlib.pyplot as plt
import seaborn as sns

# Path handling for amino
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

try:
    try:
        import amino_fast_mod as amino
        print("Successfully imported Optimized AMINO (Parallel)")
    except ImportError:
        import amino
        print("Successfully imported AMINO from:", amino.__file__)
except ImportError as e:
    print(f"FAILED to find amino.py. Error: {e}")
    amino = None

# Import refactored helpers
from data_access import create_dataframe_factory, get_feature_cols, METADATA_COLS

def compute_sequential_fisher(df_iterator_factory, target_col):
    """
    Calculates Fisher scores using a two-pass accumulation method.
    Fisher score formula: $$F = \frac{\sum n_i (\mu_i - \mu_{total})^2}{\sum n_i \sigma_i^2}$$
    """
    print("Calculating Fisher scores sequentially...")
    
    stats = {} # {feature: {class_label: [count, sum, sum_sq]}}
    global_sum = {}
    global_sum_sq = {}
    total_count = 0
    
    feature_cols = None
    
    for chunk in df_iterator_factory():
        if feature_cols is None:
            # Explicitly exclude all metadata columns and the target column
            feature_cols = [c for c in get_feature_cols(chunk) if c != target_col and c not in METADATA_COLS]
        
        y = chunk[target_col].values
        current_chunk_classes = np.unique(y)
        
        if total_count == 0:
            print(f"Found classes: {list(current_chunk_classes)}")

        for col in feature_cols:
            if col not in stats:
                stats[col] = {}
                global_sum[col] = 0.0
                global_sum_sq[col] = 0.0

            x = chunk[col].values
            global_sum[col] += np.sum(x)
            global_sum_sq[col] += np.sum(x**2)

            for cls in current_chunk_classes:
                if cls not in stats[col]:
                    stats[col][cls] = [0, 0.0, 0.0]
                
                mask = (y == cls)
                cls_data = x[mask]
                if len(cls_data) > 0:
                    stats[col][cls][0] += len(cls_data)
                    stats[col][cls][1] += np.sum(cls_data)
                    stats[col][cls][2] += np.sum(cls_data**2)
        
        total_count += len(chunk)

    fisher_scores = {}
    for col in stats:
        m_total = global_sum[col] / (total_count + 1e-12)
        numerator = 0.0
        denominator = 0.0
        
        for cls, (n_i, s_i, ss_i) in stats[col].items():
            if n_i < 2: continue
            
            m_i = s_i / n_i
            # Variance calculation: $$s^2 = \frac{\sum x^2 - \frac{(\sum x)^2}{n}}{n-1}$$
            var_i = (ss_i - (s_i**2 / n_i)) / (n_i - 1)
            
            numerator += n_i * (m_i - m_total)**2
            denominator += n_i * var_i
            
        fisher_scores[col] = numerator / (denominator + 1e-12)

    return pd.Series(fisher_scores).sort_values(ascending=False)

def extract_candidates_only(df_iterator_factory, target_col, candidates):
    """
    FIXED: Properly yields filtered chunks for concatenation.
    """
    print(f"Loading {len(candidates)} candidate features into memory...")
    
    potential_meta = [c for c in METADATA_COLS if c != target_col]
    
    def chunk_generator():
        for chunk in df_iterator_factory():
            # Identify which requested metadata actually exists in this chunk
            existing_meta = [c for c in potential_meta if c in chunk.columns]
            # Build final column list for this chunk, removing duplicates
            cols_to_extract = list(dict.fromkeys(candidates + [target_col] + existing_meta))
            cols_available = [c for c in cols_to_extract if c in chunk.columns]
            yield chunk[cols_available]

    full_df = pd.concat(chunk_generator(), ignore_index=False)
    gc.collect()
    return full_df

def visualize_amino_diagnostics(fisher_series, candidate_df, final_features, target_col):
    """Standardized diagnostic dashboard for Feature Selection results."""
    print("📊 Generating AMINO diagnostic plots...")
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        sns.set_theme(style="whitegrid")
        fig, axes = plt.subplots(1, 3, figsize=(21, 6))
        
        # 1. Signal Strength (Fisher Scree Plot)
        y_vals = fisher_series.values
        axes[0].plot(range(len(y_vals)), y_vals, color='#333333', lw=1, alpha=0.5)
        axes[0].fill_between(range(len(y_vals)), y_vals, color='#333333', alpha=0.1)
        
        # Highlight final AMINO selected features
        if len(final_features) > 0:
            selected_indices = [fisher_series.index.get_loc(f) for f in final_features if f in fisher_series.index]
            axes[0].scatter(selected_indices, fisher_series.iloc[selected_indices], color='red', s=45, label='AMINO Selected', zorder=5)
            axes[0].legend()
            
        axes[0].set_title("Feature Signal Strength (Fisher)", fontsize=14)
        axes[0].set_xlabel("Feature Rank")
        axes[0].set_ylabel("Fisher Score")
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

def run_fisher_amino_pipeline(df_iterator_factory, target_col='class', max_outputs=5, knee_S=1.0):
    fisher_s = compute_sequential_fisher(df_iterator_factory, target_col)
    
    # Knee Detection
    y_vals = fisher_s.values
    kn = KneeLocator(range(1, len(y_vals) + 1), y_vals, curve='convex', direction='decreasing', S=knee_S)
    threshold = kn.knee_y if kn.knee_y else np.percentile(y_vals, 90)
    candidate_features = fisher_s[fisher_s >= threshold].index.tolist()

    # Data Extraction
    df_amino_input = extract_candidates_only(df_iterator_factory, target_col, candidate_features)

    # AMINO Reduction
    if amino and len(candidate_features) > 0:
        print(f"Running AMINO on {len(candidate_features)} candidates...")
        ops = [amino.OrderParameter(name, df_amino_input[name].values) for name in candidate_features]
        final_ops = amino.find_ops(ops, max_outputs=max_outputs)
        final_names = [getattr(op, 'name', str(op)) for op in final_ops]
    else:
        print("AMINO skipped (missing library or candidates). Keeping top candidates.")
        limit = max_outputs if max_outputs is not None else 5
        final_names = candidate_features[:limit]

    visualize_amino_diagnostics(fisher_s, df_amino_input, final_names, target_col)

    meta_present = [c for c in df_amino_input.columns if c in METADATA_COLS]
    final_columns = list(dict.fromkeys(final_names + [target_col] + meta_present))
    
    return df_amino_input[final_columns]