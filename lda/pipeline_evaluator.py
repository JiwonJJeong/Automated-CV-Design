import plotly.graph_objects as go
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from pathlib import Path
import pickle
import sys
import gc

def extract_selected_features(fs_result):
    """Extract selected feature names from different FS method result formats."""
    # Logic: Favor explicit attrs, then columns, then dict keys
    if hasattr(fs_result, 'attrs') and 'selected_features' in fs_result.attrs:
        return fs_result.attrs['selected_features']
    elif hasattr(fs_result, 'columns'):
        # Exclude metadata and target columns
        from data_access import METADATA_COLS
        exclude_cols = ['class', 'target'] + list(METADATA_COLS)
        return [c for c in fs_result.columns if c not in exclude_cols]
    elif isinstance(fs_result, dict):
        if 'data' in fs_result:
            # Recursive check on the wrapped data
            return extract_selected_features(fs_result['data'])
            
        # Try to find feature list in common dict keys
        for key in ['selected_features', 'features', 'feature_names', 'X_selected']:
            if key in fs_result:
                # If it's a dataframe under one of these keys, get columns
                val = fs_result[key]
                if hasattr(val, 'columns'):
                     from data_access import METADATA_COLS
                     exclude_cols = ['class', 'target'] + list(METADATA_COLS)
                     return [c for c in val.columns if c not in exclude_cols]
                # If it's a list, return it
                return val
    
    return []

def evaluate_separability(df, target_col='class', selected_features=None, return_individual=False):
    """
    Calculate Fisher separability score (SB/SW ratio).
    SB: Between-class scatter (how far apart class means are)
    SW: Within-class scatter (how tight the clusters are)
    """
    from data_access import METADATA_COLS
    
    if selected_features is None:
        # Auto-detect numeric features, excluding metadata
        exclude = [target_col, 'target'] + list(METADATA_COLS)
        features = [c for c in df.columns if c not in exclude and pd.api.types.is_numeric_dtype(df[c])]
    else:
        # Use only the selected features that are actually present
        features = [f for f in selected_features if f in df.columns]
    
    if not features:
        return {} if return_individual else 0
        
    y = df[target_col].values
    unique_y = np.unique(y)
    overall_mean = df[features].mean(axis=0).values
    
    # Pre-calculate class statistics for speed
    class_data = {}
    for cls in unique_y:
        df_c = df[df[target_col] == cls][features]
        class_data[cls] = {
            'n': len(df_c),
            'mean': df_c.mean(axis=0).values,
            'var_sum': ((df_c - df_c.mean(axis=0))**2).sum(axis=0).values
        }
    
    individual_scores = {}
    total_sb = 0
    total_sw = 0
    
    for i, feat in enumerate(features):
        sb_feat = 0
        sw_feat = 0
        for cls in unique_y:
            stats = class_data[cls]
            sb_feat += stats['n'] * (stats['mean'][i] - overall_mean[i])**2
            sw_feat += stats['var_sum'][i]
        
        individual_scores[feat] = sb_feat / (sw_feat + 1e-9)
        total_sb += sb_feat
        total_sw += sw_feat
    
    if return_individual:
        return individual_scores
    
    # Revert to Pooled Ratio (Total SB / Total SW) for the aggregate score
    # to maintain consistency with historical results.
    return total_sb / (total_sw + 1e-9)

def get_feature_importance(original_df, transformed_df, selected_features=None, target_col='class'):
    """Maps selected original features back to new LDs using Correlation."""
    from data_access import METADATA_COLS
    
    if selected_features is None:
        # If no selected features specified, use all numeric features
        original_numeric = original_df.select_dtypes(include=[np.number])
        original_numeric = original_numeric.drop(columns=list(METADATA_COLS), errors='ignore')
        features_to_correlate = original_numeric.columns.tolist()
    else:
        # Use only the selected features
        features_to_correlate = [f for f in selected_features if f in original_df.columns]
        original_numeric = original_df[features_to_correlate]
    
    ld_cols = [c for c in transformed_df.columns if c != target_col]
    loadings = {}
    
    for ld in ld_cols:
        # Align indices in case of row drops
        common_idx = original_numeric.index.intersection(transformed_df.index)
        corrs = original_numeric.loc[common_idx].corrwith(transformed_df.loc[common_idx, ld])
        # Get top 3 strongest original features for this component
        loadings[ld] = corrs.abs().sort_values(ascending=False).head(3).to_dict()
        
    return loadings


def plot_feature_distributions(original_df, transformed_df, pipeline_name, target_col='class', top_n=10, survivors=None):
    """
    Plot the distribution of top N important original features across different classes using Plotly.
    Interactive dropdown allows selection of features.
    """
    import plotly.graph_objects as go
    import plotly.express as px
    import numpy as np
    
    # Get feature columns (exclude metadata)
    from data_access import METADATA_COLS
    exclude_cols = ['class', 'target'] + list(METADATA_COLS)
    
    if survivors:
        feature_cols = [c for c in survivors if c in original_df.columns]
    else:
         # Filter numeric only
        numeric_cols = original_df.select_dtypes(include=[np.number]).columns
        feature_cols = [c for c in numeric_cols if c not in exclude_cols]
    
    if not feature_cols:
        return

    # 1. Identify Top Features via Variance/Correlation
    if survivors:
        top_features = feature_cols[:top_n] # Already sorted by importance if from FS pipeline
    else:
        # Simple variance ranking for speed on the subset
        inter_vars = original_df[feature_cols].var().sort_values(ascending=False)
        top_features = inter_vars.head(top_n).index.tolist()
    
    if not top_features:
        return

    # 2. Setup Plotly Figure
    fig = go.Figure()
    
    # Use string conversion for robust sorting/grouping
    classes = sorted(original_df[target_col].astype(str).unique())
    feature_indices = {} # Map feature -> (start_idx, end_idx)
    current_idx = 0
    
    # Pre-calculate colors for classes (cycling)
    colors = px.colors.qualitative.Plotly
    class_colors = {cls: colors[i % len(colors)] for i, cls in enumerate(classes)}

    for i, feat in enumerate(top_features):
        start_idx = current_idx
        
        try:
            for cls in classes:
                # Filter by class
                subset = original_df[original_df[target_col].astype(str) == cls][feat]
                # Cleanse data
                subset = subset.replace([np.inf, -np.inf], np.nan).dropna()
                
                if subset.empty:
                    continue

                # Add Histogram Trace
                fig.add_trace(go.Histogram(
                    x=subset,
                    name=cls,
                    opacity=0.6,
                    histnorm='probability density',
                    marker_color=class_colors.get(cls),
                    legendgroup=cls,
                    visible=(i==0), # Only first feature active initially
                    hovertemplate=f"<b>{cls}</b><br>Value: %{{x}}<br>Density: %{{y}}<extra></extra>"
                ))
                current_idx += 1
                
        except Exception as e:
            print(f"      ↳ Error plotting feature {feat}: {e}")
            pass
            
        feature_indices[feat] = (start_idx, current_idx)

    if current_idx == 0:
        print("      ↳ Info: No valid data found for distribution plotting.")
        return

    # 3. Create Dropdown Menu
    steps = []
    total_traces = current_idx
    
    for feat in top_features:
        if feat not in feature_indices: continue
        start, end = feature_indices[feat]
        
        if start == end: # No traces for this feature
            continue
            
        step = dict(
            method="update",
            args=[
                {"visible": [False] * total_traces},
                {"title": f"Distribution of {feat} by {target_col}"}
            ],
            label=feat
        )
        # Set visible=True for the slice belonging to this feature
        for k in range(start, end):
            step["args"][0]["visible"][k] = True
            
        steps.append(step)

    # 4. Layout
    fig.update_layout(
        title=f"Distribution of {top_features[0]} by {target_col}" if top_features else f"Feature Distributions by {target_col}",
        xaxis_title="Feature Value",
        yaxis_title="Probability Density",
        barmode='overlay',
        template="plotly_white",
        height=500,
        updatemenus=[dict(
            active=0,
            buttons=steps,
            x=1.1,
            y=1.1,
            xanchor='left',
            yanchor='top',
            bgcolor='white',
            bordercolor='lightgrey'
        )]
    )
    
    # Add dropdown label annotation
    fig.add_annotation(
        text="Select Feature:",
        x=1.1, y=1.15,
        xref="paper", yref="paper",
        showarrow=False,
        xanchor="left", yanchor="bottom",
        font=dict(size=12)
    )

    fig.show()

def plot_feature_contributions(original_df, transformed_df, pipeline_name, target_col='class', top_n=10, survivors=None):
    """
    Plot the contribution of each original feature to each latent dimension,
    ranked from highest magnitude to lowest.
    """
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import numpy as np
    
    ld_cols = [c for c in transformed_df.columns if c != target_col]
    
    if survivors:
        feature_cols = [c for c in survivors if c in original_df.columns and c not in ['construct', 'subconstruct', 'replica', 'frame_number', 'class', 'time']]
    else:
        feature_cols = [c for c in original_df.columns if c not in ['construct', 'subconstruct', 'replica', 'frame_number', 'class', 'time']]
    
    if not ld_cols or not feature_cols:
        print("      ↳ Warning: Insufficient data for feature contribution analysis.")
        return
    
    # Calculate correlations between original features and latent dimensions
    contributions = {}
    for ld_col in ld_cols:
        correlations = []
        feature_names = []
        
        for feat_col in feature_cols:
            if feat_col in original_df.columns:
                # Use common indices to ensure alignment
                common_idx = original_df.index.intersection(transformed_df.index)
                if len(common_idx) > 1:
                    orig_vals = original_df.loc[common_idx, feat_col]
                    ld_vals = transformed_df.loc[common_idx, ld_col]
                    
                    # Calculate correlation (contribution strength)
                    corr = np.corrcoef(orig_vals, ld_vals)[0, 1]
                    if not np.isnan(corr):
                        correlations.append(abs(corr))  # Use absolute value for magnitude
                        feature_names.append(feat_col)
        
        # Sort by magnitude (descending)
        sorted_indices = np.argsort(correlations)[::-1]
        contributions[ld_col] = {
            'features': [feature_names[i] for i in sorted_indices[:top_n]],
            'magnitudes': [correlations[i] for i in sorted_indices[:top_n]]
        }
    
    # Create subplots for each latent dimension
    n_dims = len(ld_cols)
    if n_dims == 0:
        return
    
    cols = min(2, n_dims)
    rows = (n_dims + cols - 1) // cols
    
    fig = make_subplots(
        rows=rows, 
        cols=cols,
        subplot_titles=[f'{ld_col} - Top {top_n} Feature Contributions' for ld_col in ld_cols],
        vertical_spacing=0.15
    )
    
    colors = px.colors.qualitative.Set3
    
    for i, ld_col in enumerate(ld_cols):
        row = (i // cols) + 1
        col = (i % cols) + 1
        
        features = contributions[ld_col]['features']
        magnitudes = contributions[ld_col]['magnitudes']
        
        fig.add_trace(
            go.Bar(
                x=magnitudes,
                y=features,
                orientation='h',
                name=ld_col,
                marker_color=colors[i % len(colors)],
                showlegend=False
            ),
            row=row, 
            col=col
        )
        
        fig.update_xaxes(title_text="Contribution Magnitude", row=row, col=col)
        fig.update_yaxes(title_text="Features", row=row, col=col)
    
    fig.update_layout(
        title=f"Feature Contributions by Latent Dimension - {pipeline_name}",
        height=400 * rows,
        template="plotly_white",
        showlegend=False
    )
    
    fig.show()

def display_hyperparameters(hyperparams, title="Hyperparameters"):
    """
    Display hyperparameters in a formatted way.
    """
    if not hyperparams:
        return
    
    print(f"      🔧 {title}:")
    if isinstance(hyperparams, dict):
        category = hyperparams.get('category', 'Unknown')
        subcategory = hyperparams.get('subcategory', 'Unknown')
        params = hyperparams.get('parameters', {})
        timestamp = hyperparams.get('timestamp', '')
        
        print(f"         Method: {category.upper()} ➔ {subcategory.upper()}")
        if timestamp:
            print(f"         Time: {timestamp}")
        
        for param_name, param_value in params.items():
            print(f"         • {param_name}: {param_value}")
    print()

def summarize_and_evaluate(results, variance_df, original_df=None, corr_threshold=0.5):
    if not results:
        print("⚠️ No results available to evaluate.")
        return

    print("\n" + "🏆 DIMENSIONALITY REDUCTION LEADERBOARD 🏆")
    print("=" * 100)
    print(f"{'Rank':<5} | {'Pipeline Name':<25} | {'Fisher Score':<12} | {'Dims'}")
    print("-" * 100)

    scored_list = []
    for name, res in results.items():
        # Handle both Dict (pipeline result) and raw DataFrame (old cache)
        if isinstance(res, dict) and 'data' in res:
            df = res['data']
            # Recover features from the dict or the dataframe attributes (MPSO)
            selected_features = res.get('selected_features', [])
            if not selected_features and hasattr(df, 'attrs') and 'selected_features' in df.attrs:
                selected_features = df.attrs['selected_features']
        else:
            df = res
            selected_features = getattr(df, 'attrs', {}).get('selected_features', [])
            
        # Fallback: if selected_features are missing (old cache), try to infer from fs method name
        if not selected_features:
            fs_method = name.split('_to_')[0]
            fs_cache_path = Path("pipeline_cache") / f"{fs_method}.pkl"
            if fs_cache_path.exists():
                try:
                    with open(fs_cache_path, 'rb') as f:
                        fs_obj = pickle.load(f)
                        selected_features = extract_selected_features(fs_obj)
                except:
                    pass

        # We evaluate the quality of the TRANSFORMATION
        score = evaluate_separability(df)
        scored_list.append({
            'name': name, 
            'score': score, 
            'df': df,
            'selected_features': selected_features
        })

    scored_list.sort(key=lambda x: x['score'], reverse=True)

    for rank, item in enumerate(scored_list, 1):
        name, total_score, df = item['name'], item['score'], item['df']
        used_features = item['selected_features']
        ld_cols = [c for c in df.columns if c != 'class']
        
        # Verify used_features against original_df to ensure they are numeric and exist
        used_features = [f for f in used_features if f in original_df.columns and pd.api.types.is_numeric_dtype(original_df[f])]
        print(f"{rank:<5} | {name:<25} | {total_score:<12.4f} | {len(ld_cols)}")
        print(f"      📉 Selection Stats: {len(used_features)} numeric features available for mapping.")
        
        # Display hyperparameters if available (Top 3 only)
        if hasattr(results[name], 'get') and rank <= 3:
            fs_hyperparams = results[name].get('feature_selection_hyperparameters')
            dr_hyperparams = results[name].get('dimensionality_reduction_hyperparameters')
            
            if fs_hyperparams:
                display_hyperparameters(fs_hyperparams, "Feature Selection Hyperparameters")
            if dr_hyperparams:
                display_hyperparameters(dr_hyperparams, "Dimensionality Reduction Hyperparameters")
        
        sys.stdout.flush()

        try:
            if used_features:
                original_numeric = original_df[used_features] 
                common_idx = original_numeric.index.intersection(df.index)
                
                if len(common_idx) < 10:
                    print(f"      ↳ Warning: Low index overlap ({len(common_idx)} rows). Check for filter mismatch.")
                
                # Calculate per-component separability
                comp_scores = evaluate_separability(df, target_col='class', return_individual=True)
                total_comp_score = sum(comp_scores.values()) + 1e-9
                
                for ld in ld_cols:
                    if ld not in df.columns: continue
                    
                    # 1. Separability contribution %
                    c_score = comp_scores.get(ld, 0)
                    c_pct = (c_score / total_comp_score) * 100
                    
                    # 2. Top 5 features by absolute correlation
                    corrs = original_numeric.loc[common_idx].corrwith(df.loc[common_idx, ld])
                    top_5 = corrs.abs().sort_values(ascending=False).head(5)
                    
                    feat_info = ", ".join([f"{k} ({corrs[k]:.2f})" for k in top_5.index])
                    print(f"      ↳ {ld} ({c_pct:.1f}%): {feat_info}")
                sys.stdout.flush()
            else:
                print("      ↳ Info: No numeric survivors found for feature mapping.")
                sys.stdout.flush()
        except Exception as e:
            print(f"      ↳ Feature Mapping Error: {e}")
            sys.stdout.flush()
            
        if rank <= 3:
            try:
                # First, show feature contributions
                print(f"      📊 Generating Feature Contributions Analysis...")
                plot_feature_contributions(original_df, df, name, survivors=used_features)
                
                # Second, show feature distributions across classes
                print(f"      📈 Generating Feature Distributions Analysis...")
                plot_feature_distributions(original_df, df, name, survivors=used_features)
                
                # Third, show the standard visualizations using unified projector
                visualize_projection(original_df, df, name, survivors=used_features)
            except Exception as viz_e:
                print(f"      ↳ Visualization Error: {viz_e}")
                sys.stdout.flush()
        print("-" * 100)
        sys.stdout.flush()
        
        # Force matplotlib cleanup if used incorrectly elsewhere
        plt.close('all')
        gc.collect()

    gc.collect()

def visualize_projection(original_df, transformed_df, pipeline_name, target_col='class', stride=20, survivors=None):
    """
    Unified visualization for 1D, 2D, and 3D projections using Plotly.
    Maintains metadata hovering capability.
    """
    import plotly.graph_objects as go
    import plotly.express as px
    from data_access import METADATA_COLS
    
    ld_cols = [c for c in transformed_df.columns if c != target_col]
    n_dims = len(ld_cols)
    
    # Check if we have enough dimensions
    if n_dims == 0:
        return

    # Downsample for plotting performance
    df_plot = transformed_df.iloc[::stride].copy()
    
    # Attach Metadata (Strided to match)
    # Ensure correct index alignment
    common_idx = df_plot.index.intersection(original_df.index)
    df_plot = df_plot.loc[common_idx]
    
    meta_cols = [c for c in original_df.columns if c in METADATA_COLS]
    meta_subset = original_df.loc[common_idx, meta_cols]
    
    # ----------------
    # 1D Visualization
    # ----------------
    if n_dims == 1:
        col = ld_cols[0]
        # Attach metadata to df_plot mainly for hover usage in px
        for m in meta_cols:
            df_plot[m] = meta_subset[m]
            
        fig = px.strip(df_plot, x=col, y=target_col, color=target_col, 
                      hover_data=meta_cols,
                      title=f"1D Projection: {pipeline_name}",
                      template="plotly_white")
        fig.show()
        return

    # ----------------
    # 2D & 3D Setup
    # ----------------
    fig = go.Figure()

    # Create Hover Template
    hover_tmpl = "<b>Class: %{name}</b><br><br>" + \
                 "<br>".join([f"{c}: %{{customdata[{i}]}}" for i, c in enumerate(meta_cols)]) + \
                 "<extra></extra>"

    # Plot Points (Clusters)
    for cls in df_plot[target_col].unique():
        cls_mask = df_plot[target_col] == cls
        cls_data = df_plot[cls_mask]
        cls_meta = meta_subset[cls_mask]
        
        if n_dims == 2:
            fig.add_trace(go.Scatter(
                x=cls_data[ld_cols[0]], y=cls_data[ld_cols[1]],
                mode='markers', name=str(cls),
                marker=dict(size=6, opacity=0.7),
                customdata=cls_meta,
                hovertemplate=hover_tmpl
            ))
        else: # 3D or more (take first 3)
            fig.add_trace(go.Scatter3d(
                x=cls_data[ld_cols[0]], y=cls_data[ld_cols[1]], z=cls_data[ld_cols[2]],
                mode='markers', name=str(cls),
                marker=dict(size=3, opacity=0.8),
                customdata=cls_meta,
                hovertemplate=hover_tmpl
            ))

    # Plot Loadings (Arrows/Lines)
    if survivors:
        # Calculate loadings using subset for speed
        available_survivors = [c for c in survivors if c in original_df.columns]
        if not available_survivors:
             return

        df_orig_sub = original_df.loc[common_idx, available_survivors].select_dtypes(include=[np.number])
        if not df_orig_sub.empty:
            # Correlation with the plotted LDs
            dims_to_corr = ld_cols[:2] if n_dims == 2 else ld_cols[:3]
            loadings = df_orig_sub.apply(lambda x: df_plot[dims_to_corr].corrwith(x)).T
            
            # Scale arrows to match plot dimensions
            plot_max = df_plot[dims_to_corr].abs().max().max()
            scale = plot_max * 0.9
            
            # Take top features
            top_features = loadings.abs().sum(axis=1).sort_values(ascending=False).head(8).index
            
            for feat in top_features:
                vec = loadings.loc[feat] * scale
                
                if n_dims == 2:
                     fig.add_trace(go.Scatter(
                        x=[0, vec[0]], y=[0, vec[1]],
                        mode='lines+text', text=["", feat],
                        line=dict(color='red', width=2),
                        showlegend=False
                    ))
                else: # 3D
                    fig.add_trace(go.Scatter3d(
                        x=[0, vec[0]], y=[0, vec[1]], z=[0, vec[2]],
                        mode='lines+text', text=["", feat],
                        line=dict(color='red', width=4),
                        showlegend=False
                    ))

    # Layout
    if n_dims == 2:
        fig.update_layout(title=f"2D Projection: {pipeline_name}", 
                         xaxis_title=ld_cols[0], yaxis_title=ld_cols[1],
                         template="plotly_white")
    else:
        fig.update_layout(title=f"3D Projection: {pipeline_name}",
                         scene=dict(xaxis_title=ld_cols[0], yaxis_title=ld_cols[1], zaxis_title=ld_cols[2]),
                         template="plotly_white")
                         
    fig.show()

