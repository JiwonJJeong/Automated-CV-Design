import plotly.graph_objects as go
import numpy as np
import pandas as pd
from pathlib import Path
import pickle
import sys

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
        # Try to find feature list in common dict keys
        for key in ['selected_features', 'features', 'feature_names']:
            if key in fs_result:
                return fs_result[key]
    else:
        return ["Unknown format"]

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


def plot_feature_distributions(original_df, transformed_df, pipeline_name, target_col='class', top_n=10):
    """
    Plot the distribution of top N important original features across different classes.
    """
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import numpy as np
    
    # Get feature columns (exclude metadata)
    feature_cols = [c for c in original_df.columns if c not in ['construct', 'subconstruct', 'replica', 'frame_number', 'class', 'time']]
    
    if not feature_cols:
        print("      ↳ Warning: No feature columns available for distribution analysis.")
        return
    
    # Calculate feature importance using correlation with classes
    feature_importance = {}
    for feat_col in feature_cols:
        if feat_col in original_df.columns and pd.api.types.is_numeric_dtype(original_df[feat_col]):
            # Calculate correlation with target classes (one-hot encoded)
            classes = original_df[target_col].unique()
            correlations = []
            
            for class_label in classes:
                class_mask = original_df[target_col] == class_label
                if class_mask.sum() > 1:
                    feat_vals = original_df.loc[class_mask, feat_col]
                    # Use variance as a measure of discriminative power
                    if feat_vals.var() > 0:
                        correlations.append(feat_vals.var())
            
            if correlations:
                feature_importance[feat_col] = np.mean(correlations)
    
    # Get top N most important features
    sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:top_n]
    
    if not sorted_features:
        print("      ↳ Warning: No important features found for distribution analysis.")
        return
    
    top_features = [feat for feat, _ in sorted_features]
    
    # Create subplots for feature distributions
    n_features = len(top_features)
    cols = min(3, n_features)
    rows = (n_features + cols - 1) // cols
    
    fig = make_subplots(
        rows=rows, 
        cols=cols,
        subplot_titles=[f'Distribution of {feat}' for feat in top_features],
        vertical_spacing=0.15,
        horizontal_spacing=0.1
    )
    
    colors = px.colors.qualitative.Set1
    classes = original_df[target_col].unique()
    
    for i, feat in enumerate(top_features):
        row = (i // cols) + 1
        col = (i % cols) + 1
        
        for j, class_label in enumerate(classes):
            class_data = original_df[original_df[target_col] == class_label][feat]
            
            if len(class_data) > 0:
                fig.add_trace(
                    go.Histogram(
                        x=class_data,
                        name=f'{class_label}' if i == 0 else f'{class_label}_{i}',
                        opacity=0.7,
                        marker_color=colors[j % len(colors)],
                        legendgroup=f'class_{class_label}',
                        showlegend=(i == 0)  # Only show legend for first subplot
                    ),
                    row=row, 
                    col=col
                )
        
        fig.update_xaxes(title_text=feat, row=row, col=col)
        fig.update_yaxes(title_text="Count", row=row, col=col)
    
    fig.update_layout(
        title=f"Feature Distributions by Class - Top {top_n} Features - {pipeline_name}",
        height=300 * rows,
        template="plotly_white",
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    fig.show()

def plot_feature_contributions(original_df, transformed_df, pipeline_name, target_col='class', top_n=10):
    """
    Plot the contribution of each original feature to each latent dimension,
    ranked from highest magnitude to lowest.
    """
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import numpy as np
    
    ld_cols = [c for c in transformed_df.columns if c != target_col]
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
        
        # Display hyperparameters if available
        if isinstance(results[name], dict):
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
                
                # Third, show the standard visualizations
                if len(ld_cols) >= 3:
                    visualize_cluster_biplot_3d(original_df, df, name, survivors=used_features)
                elif len(ld_cols) == 2:
                    visualize_cluster_biplot(original_df, df, name, survivors=used_features)
                elif len(ld_cols) == 1:
                    visualize_cluster_1d(df, name) # Add this call!
            except Exception as viz_e:
                print(f"      ↳ Visualization Error: {viz_e}")
                sys.stdout.flush()
        print("-" * 100)
        sys.stdout.flush()

    import gc
    gc.collect()

def visualize_cluster_1d(transformed_df, pipeline_name, target_col='class'):
    import plotly.express as px
    import pandas as pd
    import numpy as np
    
    ld_cols = [c for c in transformed_df.columns if c != target_col]
    
    # Handle case where we might have 2D data but need 1D visualization
    if len(ld_cols) == 1:
        ld_col = ld_cols[0]
        fig = px.strip(transformed_df, 
                       x=ld_col, 
                       y=target_col, 
                       color=target_col,
                       title=f"1D Distribution: {pipeline_name}",
                       labels={ld_col: "Component 1"},
                       template="plotly_white")
    elif len(ld_cols) == 2:
        # If we have 2D data, create a 1D visualization using the first component
        ld_col = ld_cols[0]
        fig = px.strip(transformed_df, 
                       x=ld_col, 
                       y=target_col, 
                       color=target_col,
                       title=f"1D Distribution (Component 1): {pipeline_name}",
                       labels={ld_col: "Component 1"},
                       template="plotly_white")
    else:
        print(f"Warning: Expected 1D data for 1D visualization, got {len(ld_cols)} dimensions")
        return
    
    fig.update_layout(showlegend=False)
    fig.show()

def visualize_cluster_biplot_3d(original_df, transformed_df, pipeline_name, target_col='class', stride=20, survivors=None):
    import plotly.graph_objects as go
    ld_cols = [c for c in transformed_df.columns if c != target_col]
    centered_df = transformed_df.copy()
    centered_df[ld_cols] = transformed_df[ld_cols] - transformed_df[ld_cols].mean()
    df_plot = centered_df.iloc[::stride]
    
    fig = go.Figure()

    # 1. Plot Clusters with Metadata Hover
    # Identify metadata: only include standardized metadata columns
    meta_cols = [c for c in original_df.columns if c in METADATA_COLS]
    
    for cls in df_plot[target_col].unique():
        cls_subset = df_plot[df_plot[target_col] == cls]
        
        # Pull metadata for this subset
        common_idx = cls_subset.index.intersection(original_df.index)
        meta_subset = original_df.loc[common_idx, meta_cols]
        
        fig.add_trace(go.Scatter3d(
            x=cls_subset[ld_cols[0]], y=cls_subset[ld_cols[1]], z=cls_subset[ld_cols[2]],
            mode='markers', name=str(cls),
            marker=dict(size=3, opacity=0.8),
            customdata=meta_subset,
            hovertemplate="<b>Class: %{name}</b><br>" + 
                          "<br>".join([f"{c}: %{{customdata[{i}]}}" for i, c in enumerate(meta_cols)]) + 
                          "<extra></extra>"
        ))

    # 2. Add Feature Arrows (Loadings) - Numeric Only!
    if survivors:
        df_sampled = original_df.iloc[::stride]
        # Strict numeric enforcement
        original_numeric = df_sampled[survivors].select_dtypes(include=[np.number])
        common_idx = original_numeric.index.intersection(df_plot.index)
        
        if not original_numeric.empty:
            loadings = original_numeric.loc[common_idx].apply(
                lambda x: df_plot.loc[common_idx, ld_cols[:3]].corrwith(x)
            ).T
            
            scale = df_plot[ld_cols[:3]].abs().max().max() * 0.9
            top_features = loadings.abs().sum(axis=1).sort_values(ascending=False).head(10).index

            for feat in top_features:
                vx, vy, vz = loadings.loc[feat] * scale
                fig.add_trace(go.Scatter3d(
                    x=[0, vx], y=[0, vy], z=[0, vz],
                    mode='lines+text', text=["", feat],
                    line=dict(color='red', width=4),
                    legendgroup="arrows", showlegend=False
                ))

    fig.update_layout(
        title=f"3D Biplot: {pipeline_name}",
        scene=dict(xaxis_title=ld_cols[0], yaxis_title=ld_cols[1], zaxis_title=ld_cols[2]),
        template="plotly_white"
    )
    fig.show()

def visualize_cluster_biplot(original_df, transformed_df, pipeline_name, target_col='class', stride=20, survivors=None):
    import plotly.graph_objects as go
    ld_cols = [c for c in transformed_df.columns if c != target_col]
    centered_df = transformed_df.copy()
    centered_df[ld_cols] = transformed_df[ld_cols] - transformed_df[ld_cols].mean()
    df_plot = centered_df.iloc[::stride]
    
    fig = go.Figure()

    # 1. Plot Clusters with Metadata Hover
    meta_cols = [c for c in original_df.columns if c in METADATA_COLS]
    
    for cls in df_plot[target_col].unique():
        cls_subset = df_plot[df_plot[target_col] == cls]
        
        # Pull metadata for this subset
        common_idx = cls_subset.index.intersection(original_df.index)
        meta_subset = original_df.loc[common_idx, meta_cols]

        fig.add_trace(go.Scatter(
            x=cls_subset[ld_cols[0]], y=cls_subset[ld_cols[1]],
            mode='markers', name=str(cls),
            marker=dict(size=8, opacity=0.6),
            customdata=meta_subset,
            hovertemplate="<b>Class: %{name}</b><br>" + 
                          "<br>".join([f"{c}: %{{customdata[{i}]}}" for i, c in enumerate(meta_cols)]) + 
                          "<extra></extra>"
        ))

    # 2. Add Feature Arrows (Loadings)
    if survivors:
        df_sampled = original_df.iloc[::stride]
        original_numeric = df_sampled[survivors].select_dtypes(include=[np.number])
        common_idx = original_numeric.index.intersection(df_plot.index)
        
        if not original_numeric.empty:
            loadings = original_numeric.loc[common_idx].apply(
                lambda x: df_plot.loc[common_idx, ld_cols[:2]].corrwith(x)
            ).T
            
            scale = df_plot[ld_cols[:2]].abs().max().max() * 0.9
            top_features = loadings.abs().sum(axis=1).sort_values(ascending=False).head(10).index

            for feat in top_features:
                vx, vy = loadings.loc[feat] * scale
                fig.add_trace(go.Scatter(
                    x=[0, vx], y=[0, vy],
                    mode='lines+text', text=["", feat],
                    textposition="top center",
                    line=dict(color='red', width=2),
                    legendgroup="arrows", showlegend=False
                ))

    fig.update_layout(
        title=f"2D Biplot: {pipeline_name}",
        xaxis_title=ld_cols[0], yaxis_title=ld_cols[1],
        template="plotly_white"
    )
    fig.show()

