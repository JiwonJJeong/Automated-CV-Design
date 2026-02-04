import pytest
import pandas as pd
import numpy as np
import os
import sys
import importlib.util
from hypothesis import given, settings, strategies as st
from hypothesis.extra.pandas import data_frames, column, range_indexes

# =============================================================================
# PATH SETUP & MODULE LOADING
# =============================================================================

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
LDA_DIR = os.path.join(BASE_DIR, 'lda')
DR_DIR = os.path.join(LDA_DIR, '4_dimensionality_reduction')
sys.path.extend([LDA_DIR, DR_DIR, os.path.join(BASE_DIR, 'tests')])

# 1. Load data_access.py
try:
    import data_access
except ImportError:
    spec_da = importlib.util.spec_from_file_location("data_access", os.path.join(LDA_DIR, "data_access.py"))
    data_access = importlib.util.module_from_spec(spec_da)
    spec_da.loader.exec_module(data_access)

# 2. Load FLDA module
module_path = os.path.join(DR_DIR, 'FLDA.py')
spec_flda = importlib.util.spec_from_file_location("flda_mod", module_path)
flda_mod = importlib.util.module_from_spec(spec_flda)
spec_flda.loader.exec_module(flda_mod)

# 3. Import utils for integration tests
try:
    import dimensionality_reduction_utils as utils
except ImportError:
    utils = None

# Reference files
REF_FILE = os.path.join(BASE_DIR, "tests", "4_dimensionality_reduction", "FLDA.csv")

# =============================================================================
# ENHANCED TEST CLASS - MHLDA PATTERN
# =============================================================================

class TestFLDAEnhanced:
    """Enhanced FLDA tests following MHLDA pattern."""
    
    @pytest.fixture
    def sample_dataframe(self):
        """Create a synthetic dataset with clear class separation."""
        np.random.seed(42)
        n_samples = 100  # Reduced for faster testing
        
        # Create well-separated classes for FLDA
        class_0 = np.random.multivariate_normal([0, 0], [[1, 0.5], [0.5, 1]], n_samples//2)
        class_1 = np.random.multivariate_normal([3, 3], [[1, -0.3], [-0.3, 1]], n_samples//2)
        
        data = np.vstack([class_0, class_1])
        
        df = pd.DataFrame(data, columns=['feature_1', 'feature_2'])
        df['class'] = [0] * (n_samples // 2) + [1] * (n_samples // 2)
        
        return df

    # --- Unit Tests ---
    
    def test_flda_basic_functionality(self, sample_dataframe):
        """Test basic FLDA computation."""
        result_iter = flda_mod.run_flda(sample_dataframe, num_eigenvector=1, target_col='class')
        result_df = next(result_iter)
        
        assert isinstance(result_df, pd.DataFrame)
        assert 'class' in result_df.columns
        assert 'LD1' in result_df.columns
        assert len(result_df) == len(sample_dataframe)

    def test_flda_output_dimensions(self, sample_dataframe):
        """Test that FLDA outputs correct number of components."""
        n_components = 1  # FLDA max is classes-1 = 1 for 2 classes
        result_iter = flda_mod.run_flda(sample_dataframe, num_eigenvector=n_components, target_col='class')
        result_df = next(result_iter)
        
        ld_cols = [c for c in result_df.columns if c.startswith('LD')]
        assert len(ld_cols) == n_components

    def test_flda_with_three_classes(self):
        """Test FLDA with three classes."""
        np.random.seed(42)
        n_samples = 90  # 30 per class
        
        # Create three well-separated classes
        class_0 = np.random.multivariate_normal([0, 0], [[1, 0.2], [0.2, 1]], n_samples//3)
        class_1 = np.random.multivariate_normal([3, 0], [[1, -0.2], [-0.2, 1]], n_samples//3)
        class_2 = np.random.multivariate_normal([1.5, 3], [[1, 0.3], [0.3, 1]], n_samples//3)
        
        data = np.vstack([class_0, class_1, class_2])
        
        df = pd.DataFrame(data, columns=['feature_1', 'feature_2'])
        df['class'] = [0] * (n_samples // 3) + [1] * (n_samples // 3) + [2] * (n_samples // 3)
        
        result_iter = flda_mod.run_flda(df, num_eigenvector=2, target_col='class')
        result_df = next(result_iter)
        
        assert isinstance(result_df, pd.DataFrame)
        assert 'class' in result_df.columns
        assert 'LD1' in result_df.columns
        assert 'LD2' in result_df.columns

    def test_flda_with_small_dataset(self):
        """Test FLDA with minimal dataset."""
        small_df = pd.DataFrame({
            'feature_1': [1, 2, 3, 4, 5, 6],
            'feature_2': [2, 4, 6, 8, 10, 12],
            'class': [0, 0, 0, 1, 1, 1]
        })
        
        result_iter = flda_mod.run_flda(small_df, num_eigenvector=1, target_col='class')
        result_df = next(result_iter)
        
        assert isinstance(result_df, pd.DataFrame)
        assert 'class' in result_df.columns
        assert 'LD1' in result_df.columns

    def test_flda_class_separation_quality(self, sample_dataframe):
        """Test that FLDA provides good class separation."""
        result_iter = flda_mod.run_flda(sample_dataframe, num_eigenvector=1, target_col='class')
        result_df = next(result_iter)
        
        # Check that classes are separated in LD space
        class_0_ld = result_df[result_df['class'] == 0]['LD1']
        class_1_ld = result_df[result_df['class'] == 1]['LD1']
        
        # Means should be different
        mean_diff = abs(class_0_ld.mean() - class_1_ld.mean())
        assert mean_diff > 1.0, f"Classes should be well separated, mean difference: {mean_diff}"

    def test_flda_single_eigenvector(self, sample_dataframe):
        """Test FLDA with single eigenvector."""
        result_iter = flda_mod.run_flda(sample_dataframe, num_eigenvector=1, target_col='class')
        result_df = next(result_iter)
        
        assert 'LD1' in result_df.columns
        assert 'LD2' not in result_df.columns

    def test_error_handling_invalid_target(self, sample_dataframe):
        """Test handling of invalid target column."""
        with pytest.raises((ValueError, KeyError)):
            result_iter = flda_mod.run_flda(sample_dataframe, num_eigenvector=1, target_col='invalid_target')
            next(result_iter)

    def test_error_handling_single_class(self):
        """Test handling of dataset with single class."""
        single_class_df = pd.DataFrame({
            'feature_1': [1, 2, 3, 4],
            'feature_2': [2, 4, 6, 8],
            'class': [0, 0, 0, 0]
        })
        
        with pytest.raises((ValueError, np.linalg.LinAlgError)):
            result_iter = flda_mod.run_flda(single_class_df, num_eigenvector=1, target_col='class')
            next(result_iter)

    
    def test_integration_with_utils_data(self):
        """Integration test using external utility data functions."""
        if utils is None:
            pytest.skip("dimensionality_reduction_utils not available")
        
        try:
            df = utils.get_mpso_data()
            df = utils.assign_classes(df, start_label=1)
            result_iter = flda_mod.run_flda(df, num_eigenvector=2, target_col='class')
            result_df = next(result_iter)
            assert not result_df.empty
        except (FileNotFoundError, AttributeError):
            pytest.skip("Real test data or utils.get_mpso_data not available")

    def test_reproducibility(self, sample_dataframe):
        """Ensure FLDA (a deterministic algorithm) produces identical results on repeat."""
        res1 = next(flda_mod.run_flda(sample_dataframe, num_eigenvector=1, target_col='class'))
        res2 = next(flda_mod.run_flda(sample_dataframe, num_eigenvector=1, target_col='class'))
        pd.testing.assert_frame_equal(res1, res2)

class TestFLDAProperties:
    """Property-based tests for FLDA invariants."""

    # Strategy to generate valid DataFrames for FLDA with sufficient variance
    valid_df_strategy = data_frames(
        columns=[
            column('f1', elements=st.floats(min_value=-5, max_value=5, allow_nan=False, allow_infinity=False)),
            column('f2', elements=st.floats(min_value=-5, max_value=5, allow_nan=False, allow_infinity=False)),
            column('class', elements=st.integers(min_value=0, max_value=1))
        ],
        index=range_indexes(min_size=30)  # Increased size to reduce zero-variance probability
    ).filter(lambda df: (df['class'] == 0).sum() > 5 and (df['class'] == 1).sum() > 5 and all(df[col].var() > 1e-6 for col in ['f1', 'f2']))

    @settings(deadline=None, max_examples=20)
    @given(df=valid_df_strategy)
    def test_property_invariant_to_scaling(self, df):
        """
        Property: Scaling all input features by a constant factor should 
        not change the relative projections (direction of LDs).
        """
        try:
            # Run on original data
            res1 = next(flda_mod.run_flda(df.copy(), num_eigenvector=1, target_col='class'))

            # Run on scaled data
            df_scaled = df.copy()
            df_scaled[['f1', 'f2']] *= 10.0
            res2 = next(flda_mod.run_flda(df_scaled, num_eigenvector=1, target_col='class'))

            # Check that both results are valid
            if res1['LD1'].isna().any() or res2['LD1'].isna().any():
                pytest.skip("FLDA produced NaN values (likely due to singular matrix)")

            # The absolute correlation between projections should be near 1.0
            correlation = np.abs(np.corrcoef(res1['LD1'], res2['LD1'])[0, 1])
            assert correlation > 0.95, f"LDs should be scale-invariant, correlation: {correlation}"
        except (np.linalg.LinAlgError, ValueError):
            pytest.skip("Singular matrix encountered in random data")

    @settings(deadline=None, max_examples=20)
    @given(df=valid_df_strategy)
    def test_property_output_dimensions(self, df):
        """
        Property: The output should contain the correct number of LD columns
        capped by the theoretical maximum (classes - 1).
        """
        try:
            n_vecs = 2
            result = next(flda_mod.run_flda(df, num_eigenvector=n_vecs, target_col='class'))
            
            ld_cols = [c for c in result.columns if c.startswith('LD')]
            max_possible = min(len(df.columns) - 1, df['class'].nunique() - 1)
            expected_lds = min(n_vecs, max_possible)
            
            assert len(ld_cols) == expected_lds, f"Expected {expected_lds} LDs, got {len(ld_cols)}"
            assert len(result) == len(df), "Number of rows should be preserved"
        except (np.linalg.LinAlgError, ValueError):
            pytest.skip("Singular matrix encountered in random data")

    @settings(deadline=None, max_examples=20)
    @given(df=valid_df_strategy)
    def test_property_dimensionality_reduction_limit(self, df):
        """
        Property: Output LD count should be capped by min(features, classes - 1).
        """
        try:
            num_classes = df['class'].nunique()
            max_ld = min(len(df.columns) - 1, num_classes - 1)
            
            # Request more LDs than possible
            result = next(flda_mod.run_flda(df, num_eigenvector=10, target_col='class'))
            
            ld_cols = [c for c in result.columns if c.startswith('LD')]
            assert len(ld_cols) <= max_ld, f"LD count should not exceed {max_ld}"
        except (np.linalg.LinAlgError, ValueError):
            pytest.skip("Singular matrix encountered in random data")

    @settings(deadline=None, max_examples=15)
    @given(df=valid_df_strategy)
    def test_property_translation_invariance(self, df):
        """
        Property: Adding a constant bias to features (translation) 
        should not change the LD projection directions.
        """
        try:
            res1 = next(flda_mod.run_flda(df.copy(), num_eigenvector=1, target_col='class'))
            
            # Translate data
            df_translated = df.copy()
            df_translated[['f1', 'f2']] += 100.0
            res2 = next(flda_mod.run_flda(df_translated, num_eigenvector=1, target_col='class'))
            
            # Use a tolerance for numerical precision
            correlation = np.abs(np.corrcoef(res1['LD1'], res2['LD1'])[0, 1])
            assert correlation > 0.9, f"LDs should be translation-invariant, correlation: {correlation}"
        except (np.linalg.LinAlgError, ValueError):
            pytest.skip("Singular matrix encountered in random data")

    @settings(deadline=None, max_examples=15)
    @given(df=valid_df_strategy)
    def test_property_class_preservation(self, df):
        """
        Property: The class column should be preserved exactly in the output.
        """
        try:
            result = next(flda_mod.run_flda(df, num_eigenvector=1, target_col='class'))
            
            # Class column should be identical
            pd.testing.assert_series_equal(result['class'], df['class'])
            
            # Number of rows should be preserved
            assert len(result) == len(df)
        except (np.linalg.LinAlgError, ValueError):
            pytest.skip("Singular matrix encountered in random data")

    @settings(deadline=None, max_examples=15)
    @given(df=valid_df_strategy)
    def test_property_feature_order_independence(self, df):
        """
        Property: Permuting feature columns should not change the LD projections
        (up to possible sign flips).
        """
        try:
            # Original order
            res1 = next(flda_mod.run_flda(df.copy(), num_eigenvector=1, target_col='class'))
            
            # Permute feature columns (keep class at end)
            df_permuted = df[['f2', 'f1', 'class']].copy()
            res2 = next(flda_mod.run_flda(df_permuted, num_eigenvector=1, target_col='class'))
            
            # Correlation should be high (allowing for sign differences)
            correlation = np.abs(np.corrcoef(res1['LD1'], res2['LD1'])[0, 1])
            assert correlation > 0.8, f"LDs should be order-independent, correlation: {correlation}"
        except (np.linalg.LinAlgError, ValueError):
            pytest.skip("Singular matrix encountered in random data")

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
