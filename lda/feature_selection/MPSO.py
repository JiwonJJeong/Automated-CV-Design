import numpy as np
import pandas as pd
import gc
from sklearn.model_selection import cross_val_score
from sklearn.svm import LinearSVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.ensemble import BaggingClassifier
from niapy.problems import Problem
from niapy.task import Task
from niapy.algorithms.basic import ParticleSwarmOptimization
from ..feature_scaling.standard import create_standard_scaled_generator

# Standardized helpers
from data_access import get_feature_cols, METADATA_COLS

# --- PASS 1: STREAMING FISHER ---

def compute_fisher_scores(df_iterator, target_col='class', stride=1):
    """Memory-efficient Fisher scores with stride support."""
    print(f"Pass 1: Computing Fisher scores (stride={stride})...")
    stats = {}
    feature_cols = None
    total_n = 0

    for chunk in df_iterator:
        # APPLY STRIDE TO PASS 1
        if stride > 1:
            chunk = chunk.iloc[::stride]
            
        if feature_cols is None:
            # Explicitly exclude metadata and target column
            feature_cols = [c for c in get_feature_cols(chunk) 
                           if c != target_col and c not in METADATA_COLS and pd.api.types.is_numeric_dtype(chunk[c])]
        
        y = chunk[target_col].values
        for label in np.unique(y):
            mask = (y == label)
            data = chunk.loc[mask, feature_cols].values
            
            if label not in stats:
                stats[label] = {'n': 0, 'sum': 0.0, 'sum_sq': 0.0}
            
            stats[label]['n'] += data.shape[0]
            stats[label]['sum'] += np.sum(data, axis=0)
            stats[label]['sum_sq'] += np.sum(data**2, axis=0)
        
        total_n += len(chunk)

    if total_n == 0:
        return pd.Series(dtype=float)

    global_mean = sum(s['sum'] for s in stats.values()) / total_n
    num, den = np.zeros(len(feature_cols)), np.zeros(len(feature_cols))
    
    for s in stats.values():
        m_k = s['sum'] / s['n']
        ss_within = s['sum_sq'] - (s['sum']**2 / s['n'])
        num += s['n'] * (m_k - global_mean)**2
        den += ss_within
    
    scores = num / (den + 1e-12)
    return pd.Series(scores, index=feature_cols).sort_values(ascending=False)

# --- MPSO PROBLEM DEFINITION ---

class MPSOProjectionProblem(Problem):
    def __init__(self, X_train, y_train, dims, alpha=0.9, threshold=0.5, n_estimators=5, cv=3, redundancy_weight=0.2):
        super().__init__(dimension=X_train.shape[1] * dims, lower=0, upper=1)
        self.X_train, self.y_train = X_train, y_train
        self.dims = dims
        self.alpha = alpha
        self.threshold = threshold
        self.n_estimators = n_estimators
        self.cv = cv
        self.redundancy_weight = redundancy_weight
        self.n_feats = X_train.shape[1]
        self.eval_count = 0  # Track progress

    def _evaluate(self, x):
        self.eval_count += 1
        
        # 1. Use Soft Selection (or keep your binary, but scaling is key)
        # Reshape into (Features, Dims)
        weights = x.reshape((self.n_feats, self.dims))
        sel_matrix = (weights > self.threshold).astype(float)
        
        if np.any(np.sum(sel_matrix, axis=0) == 0):
            return 1.0

        # 2. Projection with Scaling (CRITICAL)
        # Ensure features are zero-mean/unit-variance so one doesn't dominate
        # Ideally, X_train should be pre-scaled outside the loop.
        projected = np.matmul(self.X_train, sel_matrix) 
        # Normalize by count to maintain scale
        projected /= (np.sum(sel_matrix, axis=0) + 1e-12)

        # 3. Accuracy Calculation
        clf = OneVsRestClassifier(BaggingClassifier(LinearSVC(dual=False, tol=1e-3), n_estimators=self.n_estimators))
        acc = cross_val_score(clf, projected, self.y_train, cv=self.cv).mean()

        # 4. Enhanced Redundancy (r-squared)
        if self.dims > 1:
            # Add noise to prevent NaNs in corrcoef
            noise = np.random.normal(0, 1e-10, projected.shape)
            corr_matrix = np.corrcoef(projected + noise, rowvar=False)
            # Use R-squared to punish high correlations exponentially
            r_sq_matrix = np.square(corr_matrix)
            redundancy = (np.sum(r_sq_matrix) - self.dims) / (self.dims * (self.dims - 1))
            
            # 5. Feature Overlap Penalty (New)
            # Penalize if the same feature is used in multiple columns
            feature_usage = np.sum(sel_matrix, axis=1) # How many dims use feature i
            overlap = np.sum(feature_usage[feature_usage > 1]) / (self.n_feats * self.dims)
        else:
            redundancy = 0
            overlap = 0
            
        sparsity = np.sum(sel_matrix) / (self.n_feats * self.dims)

        # 6. Unified Fitness
        # Adjust weights: Try increasing redundancy_weight to 0.4 or 0.5
        error_term = self.alpha * (1 - acc) + (1 - self.alpha) * sparsity
        
        # Mix in redundancy and overlap
        total_redundancy = (0.7 * redundancy) + (0.3 * overlap)
        
        fitness = (1.0 - self.redundancy_weight) * error_term + self.redundancy_weight * total_redundancy
        return fitness

# --- MAIN PIPELINE ---
def run_mpso_pipeline(df_iterator_factory, target_col='class', dims=5, candidate_limit=250, max_iter=10, 
                        alpha=0.9, threshold=0.5, redundancy_weight=0.2, population_size=None, stride=1, knee_S=2.0, **kwargs):
    """
    Integrated Pipeline: Optimizes on strided data, returns FULL projected dataset.
    Advanced parameters (pop_scaling, min_pop, max_pop, n_estimators, cv, seed) 
    can be passed via kwargs.
    """
    # Apply standard scaling to the data
    scaled_df_factory = create_standard_scaled_generator(df_iterator_factory)
    
    # 1. Pass 1: Filter (Strided)
    fisher_scores = compute_fisher_scores(df_iterator_factory(), target_col, stride=stride)
    if fisher_scores.empty:
        return pd.DataFrame()
        
    if candidate_limit is None:
        from kneed import KneeLocator
        y_vals = fisher_scores.values
        kn = KneeLocator(range(len(y_vals)), y_vals, curve='convex', direction='decreasing', S=knee_S)
        cutoff_idx = kn.knee if kn.knee is not None else min(250, len(y_vals))
        candidates = fisher_scores.index[:cutoff_idx+1].tolist()
        print(f"Dynamic candidate selection: Fisher score knee at index {cutoff_idx} ({len(candidates)} features)")
    else:
        candidates = fisher_scores.index[:candidate_limit].tolist()

    if dims is None:
        dims = 5
        print(f"MPSO: Output dimensions set to default (dims={dims})")
    
    # 2. Pass 2: Selective RAM Load (Strided for MPSO)
    print(f"Pass 2: Loading search data (stride={stride})...")
    search_data = []
    meta_to_keep = [c for c in METADATA_COLS if c != target_col]
    
    for chunk in df_iterator_factory():
        if stride > 1:
            chunk = chunk.iloc[::stride]
        # Use dict.fromkeys to preserve order while removing duplicates (e.g., if target_col is in candidates or meta)
        cols_to_extract = list(dict.fromkeys(candidates + [target_col] + meta_to_keep))
        available = [c for c in cols_to_extract if c in chunk.columns]
        search_data.append(chunk[available])
    
    temp_df = pd.concat(search_data, ignore_index=False)
    X_search = temp_df[candidates].values
    y_search = temp_df[target_col].values
    
    del search_data
    gc.collect()

    # 3. Particle Swarm Optimization
    if population_size is None:
        dynamic_pop = int(np.clip(len(candidates) * dims * pop_scaling, min_pop, max_pop))
    else:
        dynamic_pop = population_size
        
    print(f"Beginning Swarm Optimization on {X_search.shape[0]} samples...")
    problem = MPSOProjectionProblem(X_search, y_search, dims=dims, alpha=alpha, threshold=threshold, 
                                     n_estimators=n_estimators, cv=cv, redundancy_weight=redundancy_weight)
    task = Task(problem, max_iters=max_iter)
    algorithm = ParticleSwarmOptimization(population_size=dynamic_pop, seed=seed)
    
    best_x, _ = algorithm.run(task)
    print(f"\n✅ Optimization complete.")
    
    # Generate the "Recipe" matrix and Column Names ONCE
    best_x_reshaped = best_x.reshape((len(candidates), dims))
    final_sel = (best_x_reshaped > threshold)
    projection_weights = np.sum(final_sel, axis=0) + 1e-12

    dim_columns = []
    for dim_idx in range(dims):
        selected_in_dim = np.where(final_sel[:, dim_idx])[0]
        if len(selected_in_dim) > 0:
            dim_scores = best_x_reshaped[selected_in_dim, dim_idx]
            # Take up to top 2 contributing features for the name
            top_local_idx = np.argsort(dim_scores)[-2:]
            top_features = [candidates[selected_in_dim[i]] for i in top_local_idx]
            dim_name = "_".join(top_features)
        else:
            dim_name = f"dim{dim_idx}"
        dim_columns.append(f'MPSO_{dim_name}')

    # --- PASS 3: Apply to FULL Dataset ---
    print("Pass 3: Recovering all rows and applying projection...")
    full_results = []
    
    for chunk in df_iterator_factory():
        # Ensure only available candidates are used (safeguard)
        valid_candidates = [c for c in candidates if c in chunk.columns]
        if len(valid_candidates) != len(candidates):
             # This should not happen if Pass 1/2 were consistent
             raise ValueError("Mismatch between candidate features and available columns in FULL pass.")
             
        X_chunk = chunk[candidates].values
        projected_chunk = np.matmul(X_chunk, final_sel) / projection_weights
        
        res_chunk = pd.DataFrame(
            projected_chunk, 
            columns=dim_columns,
            index=chunk.index
        )
        
        res_chunk[target_col] = chunk[target_col].values
        current_meta = [c for c in meta_to_keep if c in chunk.columns]
        for meta in current_meta:
            res_chunk[meta] = chunk[meta].values
        
        full_results.append(res_chunk)

    final_df = pd.concat(full_results, ignore_index=False)
    gc.collect()

    # --- STANDARDIZED VISUALIZATION ---
    visualize_mpso_diagnostics(final_df, fisher_scores, candidates, final_sel, target_col)

    # Identify which original features actually contributed (safeguard)
    contributing_mask = np.any(final_sel, axis=1)
    final_df.attrs['selected_features'] = [candidates[i] for i, m in enumerate(contributing_mask) if m]

    return final_df

def visualize_mpso_diagnostics(final_df, fisher_scores, candidates, final_sel, target_col):
    """Separate visualization function for MPSO diagnostics."""
    print("📊 Generating MPSO diagnostic plots...")
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        
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
            axes[1].text(0.5, 0.5, "Need 2+ Dims\nfor Heatmap", ha='center')
            
        # 3. State Space Mapping (Projected View)
        if len(proj_cols) >= 2:
            f1, f2 = proj_cols[0], proj_cols[1]
            sample_df = final_df.sample(min(2000, len(final_df)))
            sns.scatterplot(data=sample_df, x=f1, y=f2, hue=target_col, palette="deep", s=20, alpha=0.7, ax=axes[2])
            axes[2].set_title(f"MPSO State Space\n{f1} vs {f2}", fontsize=14)
        else:
            axes[2].text(0.5, 0.5, "Need 2+ Dims\nfor Scatter Plot", ha='center')
            
        plt.tight_layout()
        plt.show()
    except Exception as e:
        print(f"Visualization failed: {e}")

    # Identify which original features actually contributed (safeguard)
    contributing_mask = np.any(final_sel, axis=1)
    final_df.attrs['selected_features'] = [candidates[i] for i, m in enumerate(contributing_mask) if m]

    return final_df

if __name__ == "__main__":
    pass