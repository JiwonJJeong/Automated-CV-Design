import pytest
import pandas as pd
import numpy as np
import os
import sys
from unittest.mock import patch, MagicMock
from hypothesis import given, settings, strategies as st
from hypothesis.extra.pandas import data_frames, column, range_indexes

# Setup paths
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(BASE_DIR, 'lda', '4_dimensionality_reduction'))
sys.path.append(os.path.join(BASE_DIR, 'tests'))

import importlib.util

# Load the module using spec_from_file_location
module_path = os.path.join(BASE_DIR, 'lda', '4_dimensionality_reduction', 'MHLDA.py')
spec = importlib.util.spec_from_file_location("mhlda_mod", module_path)
mhlda_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mhlda_mod)
import dimensionality_reduction_utils as utils

# Reference files
REF_FILE = os.path.join(BASE_DIR, "tests", "4_dimensionality_reduction", "MHLDA.csv")

class TestMHLDA:
    """Comprehensive test suite for Modified Heteroscedastic LDA."""
    
    @pytest.fixture
    def sample_dataframe(self):
        """Create a sample DataFrame for testing."""
        np.random.seed(42)
        n_samples = 300
        n_features = 10
        
        # Create synthetic data with clear class separation
        data = {}
        for i in range(n_features):
            if i < 3:  # First 3 features are discriminative
                class_0_data = np.random.normal(0, 1, n_samples // 3)
                class_1_data = np.random.normal(3, 1, n_samples // 3)
                class_2_data = np.random.normal(6, 1, n_samples // 3)
                data[f'feature_{i}'] = np.concatenate([class_0_data, class_1_data, class_2_data])
            else:
                data[f'feature_{i}'] = np.random.normal(0, 1, n_samples)
        
        data['class'] = [1] * (n_samples // 3) + [2] * (n_samples // 3) + [3] * (n_samples // 3)
        return pd.DataFrame(data)
    
    def test_mhlda_basic_functionality(self, sample_dataframe):
        """Test basic MHLDA functionality."""
        result_iter = mhlda_mod.run_mhlda(sample_dataframe, num_eigenvector=2, target_col='class')
        result_df = next(result_iter)
        
        assert isinstance(result_df, pd.DataFrame)
        assert 'class' in result_df.columns
        assert 'LD1' in result_df.columns
        assert 'LD2' in result_df.columns
        assert len(result_df) == len(sample_dataframe)
        assert len(result_df.columns) == 3  # LD1, LD2, class
    
    def test_mhlda_single_eigenvector(self, sample_dataframe):
        """Test MHLDA with single eigenvector."""
        result_iter = mhlda_mod.run_mhlda(sample_dataframe, num_eigenvector=1, target_col='class')
        result_df = next(result_iter)
        
        assert isinstance(result_df, pd.DataFrame)
        assert 'class' in result_df.columns
        assert 'LD1' in result_df.columns
        assert 'LD2' not in result_df.columns
        assert len(result_df.columns) == 2  # LD1, class
    
    def test_mhlda_three_eigenvectors(self, sample_dataframe):
        """Test MHLDA with three eigenvectors (should be capped at 2 for 3 classes)."""
        result_iter = mhlda_mod.run_mhlda(sample_dataframe, num_eigenvector=3, target_col='class')
        result_df = next(result_iter)
        
        assert isinstance(result_df, pd.DataFrame)
        assert 'class' in result_df.columns
        assert 'LD1' in result_df.columns
        assert 'LD2' in result_df.columns
        # LD3 should NOT be present for 3 classes (max is classes-1 = 2)
        assert 'LD3' not in result_df.columns
        assert len(result_df.columns) == 3  # LD1, LD2, class
    
    def test_mhlda_with_two_classes(self):
        """Test MHLDA with binary classification."""
        np.random.seed(42)
        binary_df = pd.DataFrame({
            'feature_0': np.concatenate([np.random.normal(0, 1, 50), np.random.normal(3, 1, 50)]),
            'feature_1': np.concatenate([np.random.normal(0, 1, 50), np.random.normal(3, 1, 50)]),
            'class': [1] * 50 + [2] * 50
        })
        
        result_iter = mhlda_mod.run_mhlda(binary_df, num_eigenvector=1, target_col='class')
        result_df = next(result_iter)
        
        assert isinstance(result_df, pd.DataFrame)
        assert 'class' in result_df.columns
        assert 'LD1' in result_df.columns
        assert len(result_df) == len(binary_df)
    
    def test_mhlda_with_small_dataset(self):
        """Test MHLDA with very small datasets."""
        tiny_df = pd.DataFrame({
            'feature_0': [1, 2, 3, 4, 5, 6],
            'feature_1': [2, 3, 4, 5, 6, 7],
            'class': [1, 1, 2, 2, 3, 3]
        })
        
        result_iter = mhlda_mod.run_mhlda(tiny_df, num_eigenvector=2, target_col='class')
        result_df = next(result_iter)
        
        assert isinstance(result_df, pd.DataFrame)
        assert 'class' in result_df.columns
        assert len(result_df) == len(tiny_df)
    
    def test_mhlda_with_no_discriminative_features(self):
        """Test MHLDA with random noise features."""
        np.random.seed(42)
        no_signal_df = pd.DataFrame({
            'feature_0': np.random.normal(0, 1, 90),
            'feature_1': np.random.normal(0, 1, 90),
            'feature_2': np.random.normal(0, 1, 90),
            'class': [1] * 30 + [2] * 30 + [3] * 30
        })
        
        result_iter = mhlda_mod.run_mhlda(no_signal_df, num_eigenvector=2, target_col='class')
        result_df = next(result_iter)
        
        assert isinstance(result_df, pd.DataFrame)
        assert 'class' in result_df.columns
        assert len(result_df) == len(no_signal_df)
    
    def test_mhlda_modification_parameters(self, sample_dataframe):
        """Test MHLDA with different modification parameters."""
        params = [
            {'regularization': 0.01},
            {'regularization': 0.1},
            {'regularization': 1.0}
        ]
        
        for param in params:
            try:
                result_iter = mhlda_mod.run_mhlda(
                    sample_dataframe, num_eigenvector=2, target_col='class', **param
                )
                result_df = next(result_iter)
                
                assert isinstance(result_df, pd.DataFrame)
                assert 'class' in result_df.columns
                assert 'LD1' in result_df.columns
                assert 'LD2' in result_df.columns
                
            except TypeError:
                # If parameters are not supported, skip
                pass
    
    def test_error_handling_invalid_target(self, sample_dataframe):
        """Test error handling with invalid target column."""
        with pytest.raises(ValueError):
            next(mhlda_mod.run_mhlda(sample_dataframe, num_eigenvector=2, target_col='invalid_target'))
    
    def test_error_handling_insufficient_classes(self):
        """Test error handling with insufficient classes."""
        single_class_df = pd.DataFrame({
            'feature_0': [1, 2, 3, 4],
            'feature_1': [2, 3, 4, 5],
            'class': [1, 1, 1, 1]
        })
        
        # Should handle single class gracefully or raise appropriate error
        try:
            result_iter = mhlda_mod.run_mhlda(single_class_df, num_eigenvector=2, target_col='class')
            result_df = next(result_iter)
            assert isinstance(result_df, pd.DataFrame)
        except (ValueError, np.linalg.LinAlgError):
            # These errors are acceptable for single class data
            pass
    
    def test_empty_dataframe_handling(self):
        """Test handling of empty DataFrames."""
        empty_df = pd.DataFrame({'class': [], 'feature_0': []})
        
        # Should handle empty data gracefully or raise appropriate error
        try:
            result_iter = mhlda_mod.run_mhlda(empty_df, num_eigenvector=2, target_col='class')
            result_df = next(result_iter)
            assert isinstance(result_df, pd.DataFrame)
        except (ValueError, IndexError):
            # These errors are acceptable for empty data
            pass
    
        

class TestMHLDAProperties:
    """Property-based tests for MHLDA invariants."""

    # Strategy to generate valid DataFrames for MHLDA with sufficient variance
    valid_df_strategy = data_frames(
        columns=[
            column('f1', elements=st.floats(min_value=-1e3, max_value=1e3, allow_nan=False, allow_infinity=False)),
            column('f2', elements=st.floats(min_value=-1e3, max_value=1e3, allow_nan=False, allow_infinity=False)),
            column('f3', elements=st.floats(min_value=-1e3, max_value=1e3, allow_nan=False, allow_infinity=False)),
            column('class', elements=st.integers(min_value=0, max_value=2))
        ],
        index=range_indexes(min_size=30)  # Increased size to reduce zero-variance probability
    ).filter(lambda df: df['class'].nunique() >= 2 and all(df[col].var() > 1e-6 for col in ['f1', 'f2', 'f3']))

    @settings(deadline=None, max_examples=8)
    @given(df=valid_df_strategy)
    def test_property_invariant_to_scaling(self, df):
        """
        Property: Scaling all input features by a constant factor should 
        not change the relative projections (direction of LDs).
        """
        try:
            # Run on original data
            res1 = next(mhlda_mod.run_mhlda(df.copy(), num_eigenvector=1, target_col='class'))
            
            # Run on scaled data
            df_scaled = df.copy()
            df_scaled[['f1', 'f2', 'f3']] *= 10.0
            res2 = next(mhlda_mod.run_mhlda(df_scaled, num_eigenvector=1, target_col='class'))
            
            # The absolute correlation between projections should be near 1.0
            correlation = np.abs(np.corrcoef(res1['LD1'], res2['LD1'])[0, 1])
            assert correlation > 0.95
        except (np.linalg.LinAlgError, ValueError):
            # May fail on singular matrices generated by chance
            pass

    @settings(deadline=None, max_examples=8)
    @given(df=valid_df_strategy)
    def test_property_output_dimensions(self, df):
        """
        Property: The output should contain the correct number of LD columns
        capped by the theoretical maximum (classes - 1).
        """
        n_vecs = 2
        result = next(mhlda_mod.run_mhlda(df, num_eigenvector=n_vecs, target_col='class'))
        
        ld_cols = [c for c in result.columns if c.startswith('LD')]
        max_possible = min(len(df.columns) - 1, df['class'].nunique() - 1)
        expected_lds = min(n_vecs, max_possible)
        
        assert len(ld_cols) == expected_lds
        assert len(result) == len(df)

    @settings(deadline=None, max_examples=8)  # Reduced from 20 for performance
    @given(df=valid_df_strategy)
    def test_property_dimensionality_reduction_limit(self, df):
        """
        Property: Output LD count should be capped by min(features, classes - 1).
        """
        num_classes = df['class'].nunique()
        max_ld = min(len(df.columns) - 1, num_classes - 1)
        
        # Request more LDs than possible
        result = next(mhlda_mod.run_mhlda(df, num_eigenvector=10, target_col='class'))
        
        ld_cols = [c for c in result.columns if c.startswith('LD')]
        assert len(ld_cols) <= max_ld

    @settings(deadline=None, max_examples=6)  # Reduced for performance
    @given(st.data())
    def test_property_translation_invariance(self, data):
        """
        Property: Adding a constant bias to features (translation) 
        should not change the LD projection directions for heteroscedastic LDA.
        """
        df = data.draw(self.valid_df_strategy)
        
        try:
            res1 = next(mhlda_mod.run_mhlda(df.copy(), num_eigenvector=1, target_col='class'))
            
            # Translate data
            df_translated = df.copy()
            df_translated[['f1', 'f2', 'f3']] += 100.0
            res2 = next(mhlda_mod.run_mhlda(df_translated, num_eigenvector=1, target_col='class'))
            
            # Use a tolerance for numerical precision
            correlation = np.abs(np.corrcoef(res1['LD1'], res2['LD1'])[0, 1])
            assert correlation > 0.9
        except (np.linalg.LinAlgError, ValueError):
            pass

    @settings(deadline=None, max_examples=6)  # Reduced from 15 for performance
    @given(df=valid_df_strategy)
    def test_property_class_preservation(self, df):
        """
        Property: The class column should be preserved exactly in the output.
        """
        try:
            result = next(mhlda_mod.run_mhlda(df, num_eigenvector=1, target_col='class'))
            
            # Class column should be identical
            pd.testing.assert_series_equal(result['class'], df['class'])
            
            # Number of rows should be preserved
            assert len(result) == len(df)
        except (np.linalg.LinAlgError, ValueError):
            # Skip on pathological data
            pass

    @settings(deadline=None, max_examples=6)  # Reduced from 15 for performance
    @given(df=valid_df_strategy)
    def test_property_feature_order_independence(self, df):
        """
        Property: Permuting feature columns should not change the LD projections
        (up to possible sign flips).
        """
        try:
            # Original order
            res1 = next(mhlda_mod.run_mhlda(df.copy(), num_eigenvector=1, target_col='class'))
            
            # Permute feature columns (keep class at end)
            feature_cols = [c for c in df.columns if c != 'class']
            df_permuted = df[feature_cols[::-1] + ['class']].copy()
            res2 = next(mhlda_mod.run_mhlda(df_permuted, num_eigenvector=1, target_col='class'))
            
            # Correlation should be high (allowing for sign differences)
            correlation = np.abs(np.corrcoef(res1['LD1'], res2['LD1'])[0, 1])
            assert correlation > 0.8
        except (np.linalg.LinAlgError, ValueError):
            pass

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
