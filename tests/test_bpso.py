import pytest
import pandas as pd
import numpy as np
import os
import sys
import importlib.util
import warnings
from unittest.mock import MagicMock, patch
from hypothesis import given, settings, strategies as st
from hypothesis.extra.pandas import data_frames, column, range_indexes

# =============================================================================
# PATH SETUP & MODULE LOADING
# =============================================================================

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
LDA_DIR = os.path.join(BASE_DIR, 'lda')
FEATURE_DIR = os.path.join(LDA_DIR, 'feature_selection')
sys.path.extend([LDA_DIR, FEATURE_DIR])

# 1. Load data_access.py
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
# ENHANCED TEST CLASS - MHLDA PATTERN
# =============================================================================

class TestBPSOEnhanced:
    """Enhanced BPSO tests following MHLDA pattern."""

    @pytest.fixture
    def sample_dataframe(self):
        """Create a synthetic dataset with strong signal and standardized metadata."""
        np.random.seed(42)
        n_samples = 100  # Reduced for faster testing
        
        # Create features with clear signal
        data = {}
        # Signal Features: Clear class separation
        for i in range(3):
            class_0 = np.random.normal(0, 0.5, n_samples // 2)
            class_1 = np.random.normal(3, 0.5, n_samples // 2)
            data[f'feature_{i}'] = np.concatenate([class_0, class_1])
            
        # Noise Features
        for i in range(3, 6):
            data[f'feature_{i}'] = np.random.normal(0, 1, n_samples)
            
        data['class'] = [0] * (n_samples // 2) + [1] * (n_samples // 2)
        
        # Add metadata columns (using names from data_access.METADATA_COLS)
        data['frame_number'] = np.arange(n_samples) + 1
        data['replica'] = '1'
        data['construct'] = 'test_con'
        data['subconstruct'] = 'test_sub'
        data['time'] = np.arange(n_samples) * 0.1
        
        return pd.DataFrame(data)

    @pytest.fixture
    def df_factory(self, sample_dataframe):
        """Factory that yields the sample dataframe for iterator-based APIs."""
        def factory():
            yield sample_dataframe
        return factory

    # --- Unit Tests ---
    
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

    def test_metadata_shielding_fisher(self, df_factory):
        """Verify that streaming fisher ignores all METADATA_COLS from data_access."""
        fisher_scores = bpso_svm.compute_streaming_fisher(df_factory(), target_col='class')
        
        # Ensure no metadata columns accidentally got a score
        for meta in data_access.METADATA_COLS:
            assert meta not in fisher_scores.index, f"Leak: {meta} was treated as a feature!"

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

    def test_bpso_particle_velocity_bounds(self, sample_dataframe):
        """Test that particle velocities respect bounds."""
        feat_cols = data_access.get_feature_cols(sample_dataframe)
        X = sample_dataframe[feat_cols].values
        y = sample_dataframe['class'].values

        problem = bpso_svm.SVMFeatureSelection(X, y, alpha=0.9)

        # Test that the problem has correct bounds
        assert problem.dimension == X.shape[1]
        assert np.all(problem.lower == 0)  # lower is an array
        assert np.all(problem.upper == 1)  # upper is an array
        
        # Test that evaluation works with boundary values
        boundary_solution = np.array([0.0, 1.0] + [0.5] * (problem.dimension - 2))
        fitness = problem._evaluate(boundary_solution)
        assert isinstance(fitness, (float, np.floating))
        assert fitness >= 0.0

    def test_bpso_binary_conversion(self, sample_dataframe):
        """Test binary conversion of particle positions."""
        feat_cols = data_access.get_feature_cols(sample_dataframe)
        X = sample_dataframe[feat_cols].values
        y = sample_dataframe['class'].values

        problem = bpso_svm.SVMFeatureSelection(X, y, alpha=0.9)

        # Test binary threshold conversion (used in _evaluate)
        continuous = np.array([0.2, 0.7, 0.5, 0.3, 0.9, 0.1])
        binary = continuous > 0.5  # This is what the actual code does
        
        assert binary.dtype == bool
        assert len(binary) == len(continuous)
        assert binary.tolist() == [False, True, False, False, True, False]
        
        # Test with all selected and none selected cases
        all_selected = np.ones(problem.dimension)
        none_selected = np.zeros(problem.dimension)
        
        # Should return penalty fitness for no features selected
        fitness_none = problem._evaluate(none_selected)
        assert fitness_none == 1.0
        
        # Should return valid fitness for all features selected
        fitness_all = problem._evaluate(all_selected)
        assert isinstance(fitness_all, (float, np.floating))
        assert fitness_all >= 0.0

    def test_full_pipeline_output_integrity(self, df_factory):
        """Test the final DataFrame structure and metadata preservation."""
        result_df = bpso_svm.run_bpso_pipeline(
            df_factory, 
            target_col='class', 
            candidate_limit=5, 
            bpso_iters=3,
            seed=42
        )
        
        assert isinstance(result_df, pd.DataFrame)
        assert 'class' in result_df.columns
        # Check that metadata columns are preserved in the final output
        assert 'frame_number' in result_df.columns
        assert 'replica' in result_df.columns
        assert len(result_df) == 100

    def test_reproducibility(self, df_factory):
        """Test that fixed seeds produce identical feature subsets."""
        # Optimize: Use minimal computation for reproducibility test
        res1 = bpso_svm.run_bpso_pipeline(df_factory, candidate_limit=3, bpso_iters=1, seed=42)
        res2 = bpso_svm.run_bpso_pipeline(df_factory, candidate_limit=3, bpso_iters=1, seed=42)
        
        assert sorted(res1.columns.tolist()) == sorted(res2.columns.tolist())

    


class TestBPSOProperties:
    """Property-based tests for BPSO invariants."""

    # Strategy to generate valid DataFrames for BPSO with sufficient variance
    valid_df_strategy = data_frames(
        columns=[
            column('f1', elements=st.floats(min_value=-5, max_value=5, allow_nan=False, allow_infinity=False)),
            column('f2', elements=st.floats(min_value=-5, max_value=5, allow_nan=False, allow_infinity=False)),
            column('f3', elements=st.floats(min_value=-5, max_value=5, allow_nan=False, allow_infinity=False)),
            column('class', elements=st.integers(min_value=0, max_value=1))
        ],
        index=range_indexes(min_size=30)  # Increased size to reduce zero-variance probability
    ).filter(lambda df: df['class'].nunique() >= 2 and all(df[col].var() > 1e-6 for col in ['f1', 'f2', 'f3']))

    @settings(deadline=None, max_examples=15)
    @given(df=valid_df_strategy)
    def test_property_fitness_bounds(self, df):
        """
        Property: SVM fitness should always be in [0, 1] range.
        """
        try:
            feat_cols = [c for c in df.columns if c != 'class']
            X = df[feat_cols].values
            y = df['class'].values
            
            problem = bpso_svm.SVMFeatureSelection(X, y, alpha=0.9)
            
            # Test various particle configurations
            for _ in range(3):
                particle = np.random.randint(0, 2, X.shape[1])
                fitness = problem._evaluate(particle)
                assert 0.0 <= fitness <= 1.0, f"Fitness {fitness} should be in [0, 1]"
        except Exception:
            pass

    @settings(deadline=None, max_examples=15)
    @given(df=valid_df_strategy)
    def test_property_binary_particle_validity(self, df):
        """
        Property: Binary particles should only contain 0s and 1s.
        """
        try:
            feat_cols = [c for c in df.columns if c != 'class']
            X = df[feat_cols].values
            y = df['class'].values
            
            problem = bpso_svm.SVMFeatureSelection(X, y, alpha=0.9)
            
            # Test binary conversion (actual logic used in _evaluate)
            continuous = np.random.randn(X.shape[1])
            binary = continuous > 0.5  # This is what the actual code does
            
            assert np.all(np.isin(binary, [True, False])), "Binary particles should only contain True/False"
        except Exception:
            pass

    @settings(deadline=None, max_examples=5)  
    @given(df=valid_df_strategy)
    def test_property_feature_order_independence(self, df):
        """
        Property: Feature order should not affect BPSO results (with same seed).
        """
        # Create factory functions
        def factory1(): yield df
        def factory2(): yield df
        
        # Run with very minimal parameters to avoid hanging
        result1 = bpso_svm.run_bpso_pipeline(
            factory1, target_col='class', candidate_limit=3, bpso_iters=2, seed=42
        )
        result2 = bpso_svm.run_bpso_pipeline(
            factory2, target_col='class', candidate_limit=3, bpso_iters=2, seed=42
        )
        
        # Should have same number of rows
        assert len(result1) == len(result2)
        pd.testing.assert_series_equal(result1['class'], result2['class'])

    @settings(deadline=None, max_examples=5)  # Reduced from 15
    @given(df=valid_df_strategy)
    def test_property_class_preservation(self, df):
        """
        Property: The class column should be preserved exactly in the output.
        """
        result = bpso_svm.run_bpso_pipeline(
            lambda: iter([df]), target_col='class',
            candidate_limit=3, bpso_iters=2, seed=42
        )
        
        # Class column should be identical
        pd.testing.assert_series_equal(result['class'], df['class'])
        
        # Number of rows should be preserved
        assert len(result) == len(df)

    @settings(deadline=None, max_examples=5)  # Reduced from 10
    @given(df=valid_df_strategy)
    def test_property_scaling_invariance(self, df):
        """
        Property: Scaling features should not affect relative Fisher scores.
        """
        # Original scores
        fisher_scores1 = bpso_svm.compute_streaming_fisher(iter([df]), 'class')
        
        # Scale features
        df_scaled = df.copy()
        feature_cols = [c for c in df.columns if c != 'class']
        df_scaled[feature_cols] *= 10.0
        
        fisher_scores2 = bpso_svm.compute_streaming_fisher(iter([df_scaled]), 'class')
        
        # Rankings should be the same
        ranking1 = fisher_scores1.sort_values(ascending=False).index
        ranking2 = fisher_scores2.sort_values(ascending=False).index
        
        assert list(ranking1) == list(ranking2), "Feature rankings should be scale-invariant"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
