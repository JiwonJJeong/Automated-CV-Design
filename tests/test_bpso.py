import pytest
import pandas as pd
import numpy as np
import os
import sys
import importlib.util
from unittest.mock import MagicMock, patch

# =============================================================================
# PATH SETUP & MODULE LOADING
# =============================================================================

# tests/ -> project_root/
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# Add the lda directory to the path for data_access helper
LDA_DIR = os.path.join(BASE_DIR, 'lda')
sys.path.append(LDA_DIR)

# Add specific feature selection directory
FEATURE_DIR = os.path.join(LDA_DIR, '3_feature_selection')
sys.path.append(FEATURE_DIR)

# 1. Load data_access.py (the standardized helper)
try:
    import data_access
except ImportError:
    spec_da = importlib.util.spec_from_file_location("data_access", os.path.join(LDA_DIR, "data_access.py"))
    data_access = importlib.util.module_from_spec(spec_da)
    spec_da.loader.exec_module(data_access)

# 2. Load the BPSO script (the module under test)
module_path = os.path.join(FEATURE_DIR, '3.4.BPSO.py')
spec_bpso = importlib.util.spec_from_file_location("bpso_svm", module_path)
bpso_svm = importlib.util.module_from_spec(spec_bpso)
spec_bpso.loader.exec_module(bpso_svm)

# =============================================================================
# TEST CLASS
# =============================================================================

class TestBPSO:
    """
    Comprehensive test suite for BPSO pipeline using standardized data_access logic.
    """

    @pytest.fixture
    def sample_dataframe(self):
        """Create a synthetic dataset with strong signal and standardized metadata."""
        np.random.seed(42)
        n_samples = 200
        n_features = 20
        
        data = {}
        # Signal Features: Clear class separation
        for i in range(5):
            class_0 = np.random.normal(0, 0.5, n_samples // 2)
            class_1 = np.random.normal(3, 0.5, n_samples // 2)
            data[f'feature_{i}'] = np.concatenate([class_0, class_1])
            
        # Noise Features
        for i in range(5, n_features):
            data[f'feature_{i}'] = np.random.normal(0, 1, n_samples)
            
        data['class'] = [0] * (n_samples // 2) + [1] * (n_samples // 2)
        
        # Add metadata columns (using names from data_access.METADATA_COLS)
        data['frame_number'] = np.arange(n_samples) + 1
        data['replica'] = '1'
        data['construct'] = 'test_con'
        
        return pd.DataFrame(data)

    @pytest.fixture
    def df_factory(self, sample_dataframe):
        """Standard factory mockup to test Pass 1 and Pass 2 streaming."""
        def factory():
            yield sample_dataframe.iloc[:100]
            yield sample_dataframe.iloc[100:]
        return factory

    # --- Unit Tests ---

    def test_metadata_shielding_fisher(self, df_factory):
        """Verify that streaming fisher ignores all METADATA_COLS from data_access."""
        fisher_scores = bpso_svm.compute_streaming_fisher(df_factory(), target_col='class')
        
        # Ensure no metadata columns accidentally got a score
        for meta in data_access.METADATA_COLS:
            assert meta not in fisher_scores.index, f"Leak: {meta} was treated as a feature!"

    def test_svm_fitness_logic(self, sample_dataframe):
        """Test the SVMFeatureSelection problem class directly."""
        # Use helper to extract only features
        feat_cols = data_access.get_feature_cols(sample_dataframe)
        X = sample_dataframe[feat_cols].values
        y = sample_dataframe['class'].values
        
        problem = bpso_svm.SVMFeatureSelection(X, y, alpha=0.9)
        
        # All selected
        particle_all = np.ones(X.shape[1])
        assert 0.0 <= problem._evaluate(particle_all) <= 1.0
        
        # None selected (Max penalty)
        particle_none = np.zeros(X.shape[1])
        assert problem._evaluate(particle_none) == 1.0

    # --- Integration Tests ---

    def test_pipeline_factory_reuse(self, df_factory):
        """Ensure factory is called multiple times (essential for streaming)."""
        spy_factory = MagicMock(side_effect=df_factory)
        
        bpso_svm.run_bpso_pipeline(
            spy_factory, 
            target_col='class', 
            candidate_limit=5, 
            bpso_iters=2
        )
        
        # Pipeline needs fresh data for Pass 1 (Fisher) and Pass 2 (RAM Load)
        assert spy_factory.call_count >= 2

    def test_full_pipeline_output_integrity(self, df_factory):
        """Test the final DataFrame structure and metadata preservation."""
        result_df = bpso_svm.run_bpso_pipeline(
            df_factory, 
            target_col='class', 
            candidate_limit=10, 
            bpso_iters=5,
            seed=42
        )
        
        assert isinstance(result_df, pd.DataFrame)
        assert 'class' in result_df.columns
        # Check that metadata columns are preserved in the final output
        assert 'frame_number' in result_df.columns
        assert 'replica' in result_df.columns
        assert len(result_df) == 200

    def test_reproducibility(self, df_factory):
        """Test that fixed seeds produce identical feature subsets."""
        res1 = bpso_svm.run_bpso_pipeline(df_factory, candidate_limit=5, bpso_iters=5, seed=7)
        res2 = bpso_svm.run_bpso_pipeline(df_factory, candidate_limit=5, bpso_iters=5, seed=7)
        
        assert sorted(res1.columns.tolist()) == sorted(res2.columns.tolist())

if __name__ == "__main__":
    pytest.main([__file__, "-v"])