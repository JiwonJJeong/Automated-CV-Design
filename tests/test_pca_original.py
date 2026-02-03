import pytest
import pandas as pd
import numpy as np
import subprocess
import os
import sys

# Add tests directory to path to import utils
sys.path.append(os.path.dirname(__file__))
import dimensionality_reduction_utils as utils

BASE_DIR = utils.BASE_DIR
REF_FILE = os.path.join(BASE_DIR, "tests", "4_dimensionality_reduction", "PCA.csv")

def run_in_conda(code):
    cmd = ["conda", "run", "-n", "gklab", "python", "-c", code]
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=BASE_DIR)
    if result.returncode != 0:
        print(result.stdout)
        print(result.stderr)
        raise RuntimeError(f"Conda run failed with exit code {result.returncode}")
    return result.stdout

def test_pca():
    code = f"""
import pandas as pd
import sys
import os
import importlib.util
import json

# Add project directories to path
sys.path.append(os.path.join('{BASE_DIR}', 'lda', '4_dimensionality_reduction'))
sys.path.append(os.path.join('{BASE_DIR}', 'tests'))

import dimensionality_reduction_utils as utils

def load_module_from_path(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

pca_mod = load_module_from_path("pca_mod", os.path.join('{BASE_DIR}', 'lda', '4_dimensionality_reduction', 'PCA.py'))

# Prepare data matching notebook logic
df = utils.get_mpso_data()
# PCA notebook uses 0, 1, 2 labels
df = utils.assign_classes(df, start_label=0)

try:
    # Run PCA
    result_iter = pca_mod.run_pca(df, num_eigenvector=2, target_col='class')
    result_df = next(result_iter)
    
    # Output JSON for comparison
    print(result_df.to_json(orient='split'))
except Exception as e:
    print(f"ERROR: {{e}}", file=sys.stderr)
    sys.exit(1)
"""
    output = run_in_conda(code)
    try:
        json_str = output.strip().split('\n')[-1]
        result_df = pd.read_json(json_str, orient='split')
    except Exception as e:
        print(f"Failed to parse result JSON: {e}")
        print(f"Raw output: {output}")
        raise
    
    ref_df = pd.read_csv(REF_FILE)
    
    # PCA output signs can be flipped depending on implementation/ordering,
    # but the magnitudes and class column should match.
    # Check shape
    assert result_df.shape == ref_df.shape, f"Shape mismatch: {result_df.shape} vs {ref_df.shape}"
    
    # Check class column exactly
    pd.testing.assert_series_equal(result_df['class'], ref_df['class'])
    
    # Check PC values (with sign handling)
    for col in ['PC1', 'PC2']:
        # They should either be equal or exact negatives
        diff_pos = np.abs(result_df[col] - ref_df[col]).mean()
        diff_neg = np.abs(result_df[col] + ref_df[col]).mean()
        
        assert min(diff_pos, diff_neg) < 1e-5, f"Values in {col} do not match reference (even with sign flip)"

    print("PCA test passed.")

if __name__ == "__main__":
    test_pca()
