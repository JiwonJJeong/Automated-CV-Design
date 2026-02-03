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

# Define project root: tests/ -> project_root/
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

# 2. Load the MPSO script (the module under test)
# Since the filename has dots, we must use importlib
module_path = os.path.join(FEATURE_DIR, '3.5.MPSO.py')
spec_mpso = importlib.util.spec_from_file_location("mpso_module", module_path)
mpso = importlib.util.module_from_spec(spec_mpso)
spec_mpso.loader.exec_module(mpso)

# =============================================================================
# TEST CLASS
# =============================================================================

class TestMPSO:
    """
    Comprehensive test suite for Multi-objective PSO pipeline using standardized data_access.
    """

    @pytest.fixture
    def sample_dataframe(self):
        """Create a synthetic dataset with strong signal and standardized metadata."""
        np.random.seed(42)
        n_samples = 200
        n_features = 30 # Increased to test projection space
        
        data = {}
        # Signal Features
        for i in range(5):
            class_0 = np.random.normal(0, 0.4, n_samples // 2)
            class_1 = np.random.normal(2, 0.4, n_samples // 2)
            data[f'feature_{i}'] = np.concatenate([class_0, class_1])
            
        # Noise Features
        for i in range(5, n_features):
            data[f'feature_{i}'] = np.random.normal(0, 1, n_samples)
            
        data['class'] = [0] * (n_samples // 2) + [1] * (n_samples // 2)
        
        # Standard Metadata (from data_access)
        data['frame_number'] = np.arange(n_samples) + 1
        data['replica'] = '1'
        data['time'] = np.linspace(0, 100, n_samples)
        
        return pd.DataFrame(data)

    @pytest.fixture
    def df_factory(self, sample_dataframe):
        """Standard factory mockup to test Pass 1 and Pass 2 streaming."""
        def factory():
            yield sample_dataframe.iloc[:100]
            yield sample_dataframe.iloc[100:]
        return factory

    # --- Unit Tests ---

    def test_metadata_shielding_in_mpso(self, sample_dataframe):
        """Ensure the MPSO problem class only sees features, not metadata."""
        # Use data_access to get clean features
        feat_cols = data_access.get_feature_cols(sample_dataframe)
        
        # Verify that metadata columns are strictly excluded
        for meta in data_access.METADATA_COLS:
            assert meta not in feat_cols, f"Metadata Leak: {meta} found in feature columns!"

    def test_mpso_projection_logic(self, sample_dataframe):
        """Test the Problem class evaluation logic (Matrix Projection)."""
        feat_cols = data_access.get_feature_cols(sample_dataframe)
        X = sample_dataframe[feat_cols].values
        y = sample_dataframe['class'].values
        dims = 3
        
        # Initialize Problem
        problem = mpso.MPSOProjectionProblem(X, y, dims=dims)
        
        # Particle dimension should be (n_features * dims)
        expected_dim = X.shape[1] * dims
        assert problem.dimension == expected_dim
        
        # Test evaluation with a dummy particle
        dummy_particle = np.random.rand(problem.dimension)
        fitness = problem._evaluate(dummy_particle)
        
        assert isinstance(fitness, float)
        assert 0.0 <= fitness <= 1.0

    # --- Integration Tests ---

    def test_mpso_pipeline_execution(self, df_factory):
        """Test full pipeline: Fisher -> RAM -> MPSO -> Transformed Results."""
        target_dims = 4
        result_df = mpso.run_mpso_pipeline(
            df_factory, 
            target_col='class', 
            dims=target_dims,
            candidate_limit=15, 
            mpso_iters=5,
            seed=42
        )
        
        # 1. Structure Checks
        assert isinstance(result_df, pd.DataFrame)
        assert 'class' in result_df.columns
        # The result should contain exactly target_dims plus target_col and metadata
        # MPSO typically renames columns to MPSO_Dim_N
        mpso_cols = [c for c in result_df.columns if 'MPSO_Dim' in c]
        assert len(mpso_cols) == target_dims

    def test_factory_pass_logic(self, df_factory):
        """Verify the pipeline refreshes the generator for both Pass 1 and Pass 2."""
        spy_factory = MagicMock(side_effect=df_factory)
        
        mpso.run_mpso_pipeline(
            spy_factory, 
            target_col='class', 
            dims=2,
            mpso_iters=2
        )
        
        # Needs fresh data for Fisher Scoring and the Final Matrix Projection load
        assert spy_factory.call_count >= 2

    def test_reproducibility(self, df_factory):
        """Verify that seed ensures consistent matrix projection."""
        res1 = mpso.run_mpso_pipeline(df_factory, dims=2, mpso_iters=5, seed=123)
        res2 = mpso.run_mpso_pipeline(df_factory, dims=2, mpso_iters=5, seed=123)
        
        # Column names should be identical
        assert res1.columns.tolist() == res2.columns.tolist()
        # Projected values should be near-identical
        np.testing.assert_allclose(res1['MPSO_Dim_1'].values, res2['MPSO_Dim_1'].values)

if __name__ == "__main__":
    pytest.main([__file__, "-v"])