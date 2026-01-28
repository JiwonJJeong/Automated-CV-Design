import pytest
import pandas as pd
import numpy as np
import os
import sys

# Add lda directory to path to import variance
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'lda', '2_feature_extraction')))
import variance

def calculate_pairwise_distances(df):
    """
    Replicates the logic from 2.4_variance.ipynb to calculate Euclidean distances 
    between all pairs of residues.
    """
    coord_cols = [c for c in df.columns if any(c.endswith(suffix) for suffix in ['.x', '.y', '.z'])]
    n_res = int(len(coord_cols) / 3)
    
    # Store residue base names (strip .x)
    res_bases = [coord_cols[i*3][:-2] for i in range(n_res)]
    
    dist_data = {}
    
    # Nested loop for pairwise distances (upper triangle)
    for i in range(n_res):
        resid1 = res_bases[i]
        r1x = df[f"{resid1}.x"]
        r1y = df[f"{resid1}.y"]
        r1z = df[f"{resid1}.z"]
        
        for j in range(i + 1, n_res):
            resid2 = res_bases[j]
            col_name = "{}.{}".format(resid1, resid2[3:])
            
            r2x = df[f"{resid2}.x"]
            r2y = df[f"{resid2}.y"]
            r2z = df[f"{resid2}.z"]
            
            dist_data[col_name] = np.sqrt(((r1x - r2x)**2) + ((r1y - r2y)**2) + ((r1z - r2z)**2))
            
    return pd.DataFrame(dist_data)

@pytest.fixture(scope="module")
def variance_results():
    """
    Fixture that runs the variance pipeline and returns calculated vs expected data.
    """
    input_coords = '/home/jiwonjjeong/gk-lab/Automated-CV-Design/tests/2_feature_extraction/sample_CA_coords.csv'
    expected_output = '/home/jiwonjjeong/gk-lab/Automated-CV-Design/tests/2_feature_extraction/sample_CA_post_variance.csv'
    
    if not os.path.exists(input_coords) or not os.path.exists(expected_output):
        pytest.skip("Test data files not found.")

    coords_df = pd.read_csv(input_coords)
    calculated_dist_df = calculate_pairwise_distances(coords_df)
    
    # Run variance.py logic (fixed threshold 1.71 to match notebook)
    def dist_iterator():
        yield calculated_dist_df
        
    selected_features, threshold = variance.calculate_features_with_low_variance(dist_iterator(), threshold=1.71)
    variant_df = variance.remove_low_variance_features(calculated_dist_df, selected_features)
    
    expected_df = pd.read_csv(expected_output)
    expected_features = [c for c in expected_df.columns if c != 'class']
    
    return {
        "calculated_features": selected_features,
        "expected_features": expected_features,
        "calculated_df": variant_df,
        "expected_df": expected_df
    }

def test_feature_selection_matches(variance_results):
    """
    Verify that the refactored script selects the exact same features as the original notebook.
    """
    calc = set(variance_results["calculated_features"])
    exp = set(variance_results["expected_features"])
    
    assert len(calc) == len(exp), f"Feature count mismatch: {len(calc)} vs {len(exp)}"
    assert calc == exp, f"Feature set mismatch. Missing: {exp - calc}, Extra: {calc - exp}"

def test_numeric_consistency(variance_results):
    """
    Verify that numeric values in the filtered DataFrames match within tolerance.
    """
    calc_df = variance_results["calculated_df"]
    exp_df = variance_results["expected_df"]
    common_features = variance_results["calculated_features"] # Assuming set equality from prev test
    
    # Sort columns to ensure match
    calc_subset = calc_df[common_features].sort_index(axis=1)
    exp_subset = exp_df[common_features].sort_index(axis=1)
    
    # Assert equality with tolerance for floating point variations
    pd.testing.assert_frame_equal(calc_subset, exp_subset, atol=1e-5)
