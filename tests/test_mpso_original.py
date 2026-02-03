import pytest
import pandas as pd
import os
import subprocess

# File paths
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
INPUT_FILE = os.path.join(BASE_DIR, 'tests', '2_feature_extraction', 'sample_CA_post_variance.csv')
REF_FILE = os.path.join(BASE_DIR, 'tests', '3_feature_selection', 'mpso.csv')

def run_in_conda(code):
    cmd = ["conda", "run", "-n", "gklab", "python", "-c", code]
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=BASE_DIR)
    if result.returncode != 0:
        print(result.stdout)
        print(result.stderr)
        raise RuntimeError(f"Conda run failed with exit code {result.returncode}")
    return result.stdout

def test_mpso():
    code = f"""
import pandas as pd
import sys
import os
import importlib.util
import json
import numpy as np
import random

# Set seeds for reproducibility
np.random.seed(42)
random.seed(42)

sys.path.append(os.path.join('{BASE_DIR}', 'lda', '3_feature_selection'))

def load_module_from_path(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

mpso = load_module_from_path("mpso", os.path.join('{BASE_DIR}', 'lda', '3_feature_selection', '3.5.MPSO.py'))

df = pd.read_csv('{INPUT_FILE}')
df_iter = [df]

try:
    # Use lower iterations for speed in tests
    result_df = mpso.run_mpso_pipeline(df_iter, target_col='class', dims=5, mpso_iters=20)
    # Output only the JSON to stdout for easy parsing
    print(result_df.to_json(orient='split'))
except Exception as e:
    print(f"ERROR: {{e}}", file=sys.stderr)
    sys.exit(1)
"""
    output = run_in_conda(code)
    try:
        # The JSON should be the last non-empty line of stdout
        json_str = output.strip().split('\n')[-1]
        result_df = pd.read_json(json_str, orient='split')
    except Exception as e:
        print(f"Failed to parse result JSON: {e}")
        print(f"Raw output: {output}")
        raise
    
    ref_df = pd.read_csv(REF_FILE)
    
    # Check shape
    assert result_df.shape == ref_df.shape, f"Shape mismatch: {result_df.shape} vs {ref_df.shape}"
    
    # Compare values
    pd.testing.assert_frame_equal(
        result_df.sort_index(axis=1).sort_index(),
        ref_df.sort_index(axis=1).sort_index(),
        check_dtype=False,
        atol=1e-5
    )
    print("MPSO test passed with matching values.")
