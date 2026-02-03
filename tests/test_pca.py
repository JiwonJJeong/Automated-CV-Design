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
module_path = os.path.join(BASE_DIR, 'lda', '4_dimensionality_reduction', 'PCA.py')
spec = importlib.util.spec_from_file_location("pca_mod", module_path)
pca_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pca_mod)
import dimensionality_reduction_utils as utils

# Reference files
REF_FILE = os.path.join(BASE_DIR, "tests", "4_dimensionality_reduction", "PCA.csv")


class TestPCA:
    """Comprehensive test suite for Principal Component Analysis."""
    
    @pytest.fixture
    def sample_dataframe(self):
        """Create a sample DataFrame for testing."""
        np.random.seed(42)
        n_samples = 300
        n_features = 10
        
        # Create synthetic data with some correlation structure
        data = {}
        for i in range(n_features):
            if i < 3:  # First 3 features are correlated
                base_signal = np.random.randn(n_samples)
                data[f'feature_{i}'] = base_signal + 0.1 * np.random.randn(n_samples)
            else:
                data[f'feature_{i}'] = np.random.randn(n_samples)
        
        data['class'] = [0] * (n_samples // 3) + [1] * (n_samples // 3) + [2] * (n_samples // 3)
        return pd.DataFrame(data)
    
    def test_pca_basic_functionality(self, sample_dataframe):
        """Test basic PCA functionality."""
        result_iter = pca_mod.run_pca(sample_dataframe, num_eigenvector=2, target_col='class')
        result_df = next(result_iter)
        
        assert isinstance(result_df, pd.DataFrame)
        assert 'class' in result_df.columns
        assert 'PC1' in result_df.columns
        assert 'PC2' in result_df.columns
        assert len(result_df) == len(sample_dataframe)
        assert len(result_df.columns) == 3  # PC1, PC2, class
    
    def test_pca_single_eigenvector(self, sample_dataframe):
        """Test PCA with single eigenvector."""
        result_iter = pca_mod.run_pca(sample_dataframe, num_eigenvector=1, target_col='class')
        result_df = next(result_iter)
        
        assert isinstance(result_df, pd.DataFrame)
        assert 'class' in result_df.columns
        assert 'PC1' in result_df.columns
        assert 'PC2' not in result_df.columns
        assert len(result_df.columns) == 2  # PC1, class
    
    def test_pca_three_eigenvectors(self, sample_dataframe):
        """Test PCA with three eigenvectors."""
        result_iter = pca_mod.run_pca(sample_dataframe, num_eigenvector=3, target_col='class')
        result_df = next(result_iter)
        
        assert isinstance(result_df, pd.DataFrame)
        assert 'class' in result_df.columns
        assert 'PC1' in result_df.columns
        assert 'PC2' in result_df.columns
        assert 'PC3' in result_df.columns
        assert len(result_df.columns) == 4  # PC1, PC2, PC3, class
    
    def test_pca_with_two_classes(self):
        """Test PCA with binary classification."""
        np.random.seed(42)
        binary_df = pd.DataFrame({
            'feature_0': np.random.randn(100),
            'feature_1': np.random.randn(100),
            'class': [0] * 50 + [1] * 50
        })
        
        result_iter = pca_mod.run_pca(binary_df, num_eigenvector=1, target_col='class')
        result_df = next(result_iter)
        
        assert isinstance(result_df, pd.DataFrame)
        assert 'class' in result_df.columns
        assert 'PC1' in result_df.columns
        assert len(result_df) == len(binary_df)
    
    def test_pca_with_small_dataset(self):
        """Test PCA with very small datasets."""
        tiny_df = pd.DataFrame({
            'feature_0': [1, 2, 3, 4],
            'feature_1': [2, 3, 4, 5],
            'class': [0, 0, 1, 1]
        })
        
        result_iter = pca_mod.run_pca(tiny_df, num_eigenvector=2, target_col='class')
        result_df = next(result_iter)
        
        assert isinstance(result_df, pd.DataFrame)
        assert 'class' in result_df.columns
        assert len(result_df) == len(tiny_df)
    
    def test_pca_with_uncorrelated_features(self):
        """Test PCA with uncorrelated features."""
        np.random.seed(42)
        uncorrelated_df = pd.DataFrame({
            'feature_0': np.random.randn(90),
            'feature_1': np.random.randn(90),
            'feature_2': np.random.randn(90),
            'class': [0] * 30 + [1] * 30 + [2] * 30
        })
        
        result_iter = pca_mod.run_pca(uncorrelated_df, num_eigenvector=2, target_col='class')
        result_df = next(result_iter)
        
        assert isinstance(result_df, pd.DataFrame)
        assert 'class' in result_df.columns
        assert len(result_df) == len(uncorrelated_df)
    
    def test_pca_variance_explained(self, sample_dataframe):
        """Test that PCA captures variance properly."""
        result_iter = pca_mod.run_pca(sample_dataframe, num_eigenvector=2, target_col='class')
        result_df = next(result_iter)
        
        # Extract PC components
        pc1 = result_df['PC1'].values
        pc2 = result_df['PC2'].values
        
        # PC1 should have higher variance than PC2
        assert np.var(pc1) >= np.var(pc2)
        
        # Both should have non-zero variance
        assert np.var(pc1) > 1e-6
        assert np.var(pc2) > 1e-6
    
    def test_pca_orthogonality(self, sample_dataframe):
        """Test that principal components are orthogonal."""
        result_iter = pca_mod.run_pca(sample_dataframe, num_eigenvector=2, target_col='class')
        result_df = next(result_iter)
        
        # Extract PC components
        pc1 = result_df['PC1'].values
        pc2 = result_df['PC2'].values
        
        # Calculate correlation (should be close to 0 for orthogonal components)
        correlation = np.corrcoef(pc1, pc2)[0, 1]
        
        # Should be approximately orthogonal
        assert abs(correlation) < 0.1
    
    def test_pca_centering(self, sample_dataframe):
        """Test that PCA properly centers the data."""
        result_iter = pca_mod.run_pca(sample_dataframe, num_eigenvector=2, target_col='class')
        result_df = next(result_iter)
        
        # Extract PC components
        pc1 = result_df['PC1'].values
        pc2 = result_df['PC2'].values
        
        # Components should be centered (mean close to 0)
        assert abs(np.mean(pc1)) < 1e-10
        assert abs(np.mean(pc2)) < 1e-10
    
    def test_error_handling_invalid_target(self, sample_dataframe):
        """Test error handling with invalid target column."""
        with pytest.raises(ValueError):
            next(pca_mod.run_pca(sample_dataframe, num_eigenvector=2, target_col='invalid_target'))
    
    def test_error_handling_insufficient_features(self):
        """Test error handling with insufficient features."""
        single_feature_df = pd.DataFrame({
            'feature_0': [1, 2, 3, 4],
            'class': [0, 0, 1, 1]
        })
        
        # Should handle single feature gracefully
        try:
            result_iter = pca_mod.run_pca(single_feature_df, num_eigenvector=1, target_col='class')
            result_df = next(result_iter)
            assert isinstance(result_df, pd.DataFrame)
        except ValueError:
            # Some implementations might fail with single feature
            pass
    
    def test_empty_dataframe_handling(self):
        """Test handling of empty DataFrames."""
        empty_df = pd.DataFrame({'class': [], 'feature_0': []})
        
        # Should handle empty data gracefully or raise appropriate error
        try:
            result_iter = pca_mod.run_pca(empty_df, num_eigenvector=2, target_col='class')
            result_df = next(result_iter)
            assert isinstance(result_df, pd.DataFrame)
        except (ValueError, IndexError):
            # These errors are acceptable for empty data
            pass
    
    def test_integration_with_real_data(self):
        """Integration test using real data."""
        try:
            df = utils.get_mpso_data()
            df = utils.assign_classes(df, start_label=0)  # PCA uses 0, 1, 2 labels
            
            result_iter = pca_mod.run_pca(df, num_eigenvector=2, target_col='class')
            result_df = next(result_iter)
            
            assert isinstance(result_df, pd.DataFrame)
            assert 'class' in result_df.columns
            assert 'PC1' in result_df.columns
            assert 'PC2' in result_df.columns
            assert len(result_df) == len(df)
            
        except FileNotFoundError:
            pytest.skip("Real test data not available")
    
    def test_reference_output_comparison(self):
        """Test against reference output if available."""
        if not os.path.exists(REF_FILE):
            pytest.skip("Reference file not available")
        
        try:
            df = utils.get_mpso_data()
            df = utils.assign_classes(df, start_label=0)  # PCA uses 0, 1, 2 labels
            
            result_iter = pca_mod.run_pca(df, num_eigenvector=2, target_col='class')
            result_df = next(result_iter)
            
            ref_df = pd.read_csv(REF_FILE)
            
            # Check shape
            assert result_df.shape == ref_df.shape, f"Shape mismatch: {result_df.shape} vs {ref_df.shape}"
            
            # Check class column exactly
            pd.testing.assert_series_equal(result_df['class'], ref_df['class'])
            
            # Check PC values (with sign handling)
            for col in ['PC1', 'PC2']:
                diff_pos = np.abs(result_df[col] - ref_df[col]).mean()
                diff_neg = np.abs(result_df[col] + ref_df[col]).mean()
                
                assert min(diff_pos, diff_neg) < 1e-5, f"Values in {col} do not match reference"
                
        except FileNotFoundError:
            pytest.skip("Real test data not available")
    
    def test_pca_with_correlated_data(self):
        """Test PCA with highly correlated data."""
        np.random.seed(42)
        base_signal = np.random.randn(100)
        
        correlated_df = pd.DataFrame({
            'feature_0': base_signal,
            'feature_1': base_signal + 0.1 * np.random.randn(100),
            'feature_2': base_signal + 0.1 * np.random.randn(100),
            'class': [0] * 50 + [1] * 50
        })
        
        result_iter = pca_mod.run_pca(correlated_df, num_eigenvector=2, target_col='class')
        result_df = next(result_iter)
        
        assert isinstance(result_df, pd.DataFrame)
        assert 'class' in result_df.columns
        
        # PC1 should capture most of the variance
        pc1 = result_df['PC1'].values
        pc2 = result_df['PC2'].values
        
        # With correlated data, PC1 should have much higher variance than PC2
        assert np.var(pc1) > 5 * np.var(pc2)
    
    def test_pca_with_scaled_data(self, sample_dataframe):
        """Test PCA behavior with different data scales."""
        # Create data with different scales
        scaled_df = sample_dataframe.copy()
        for col in scaled_df.columns:
            if col != 'class' and col.startswith('feature_0'):
                scaled_df[col] *= 100  # Scale up first feature
        
        result_iter = pca_mod.run_pca(scaled_df, num_eigenvector=2, target_col='class')
        result_df = next(result_iter)
        
        assert isinstance(result_df, pd.DataFrame)
        assert 'class' in result_df.columns
        assert not np.any(np.isnan(result_df[['PC1', 'PC2']].values))
    
    def test_pca_numerical_stability(self, sample_dataframe):
        """Test numerical stability with extreme values."""
        # Add some extreme values to test numerical stability
        extreme_df = sample_dataframe.copy()
        extreme_df.loc[0, 'feature_0'] = 1e6
        extreme_df.loc[1, 'feature_0'] = -1e6
        
        try:
            result_iter = pca_mod.run_pca(extreme_df, num_eigenvector=2, target_col='class')
            result_df = next(result_iter)
            
            assert isinstance(result_df, pd.DataFrame)
            assert 'class' in result_df.columns
            assert not np.any(np.isnan(result_df[['PC1', 'PC2']].values))
            
        except (ValueError, np.linalg.LinAlgError):
            # Numerical instability is acceptable for extreme values
            pass
    
    def test_pca_reproducibility(self, sample_dataframe):
        """Test that PCA produces reproducible results."""
        np.random.seed(42)
        
        result_iter1 = pca_mod.run_pca(sample_dataframe, num_eigenvector=2, target_col='class')
        result_df1 = next(result_iter1)
        
        np.random.seed(42)
        
        result_iter2 = pca_mod.run_pca(sample_dataframe, num_eigenvector=2, target_col='class')
        result_df2 = next(result_iter2)
        
        # Results should be identical (PCA is deterministic)
        pd.testing.assert_frame_equal(result_df1.sort_index(), result_df2.sort_index())
    
    def test_pca_different_class_labels(self):
        """Test PCA with different class label schemes."""
        np.random.seed(42)
        
        # Test with 1, 2, 3 labels
        df_123 = pd.DataFrame({
            'feature_0': np.random.randn(90),
            'feature_1': np.random.randn(90),
            'class': [1, 2, 3] * 30
        })
        
        result_iter = pca_mod.run_pca(df_123, num_eigenvector=2, target_col='class')
        result_df = next(result_iter)
        
        assert isinstance(result_df, pd.DataFrame)
        assert 'class' in result_df.columns
        assert len(result_df) == len(df_123)
    
    def test_pca_variance_preservation(self, sample_dataframe):
        """Test that PCA preserves total variance."""
        # Get original feature variance
        feature_cols = [col for col in sample_dataframe.columns if col != 'class']
        original_variance = sample_dataframe[feature_cols].values.var(axis=0).sum()
        
        # Get PCA components
        result_iter = pca_mod.run_pca(sample_dataframe, num_eigenvector=len(feature_cols), target_col='class')
        result_df = next(result_iter)
        
        pc_cols = [col for col in result_df.columns if col.startswith('PC')]
        pca_variance = result_df[pc_cols].values.var(axis=0).sum()
        
        # Total variance should be approximately preserved
        assert abs(original_variance - pca_variance) / original_variance < 0.01
    
    def test_pca_eigenvalue_properties(self, sample_dataframe):
        """Test properties of eigenvalues in PCA."""
        result_iter = pca_mod.run_pca(sample_dataframe, num_eigenvector=3, target_col='class')
        result_df = next(result_iter)
        
        # Extract PC components
        pc1 = result_df['PC1'].values
        pc2 = result_df['PC2'].values
        pc3 = result_df['PC3'].values
        
        # Variances should be in descending order
        assert np.var(pc1) >= np.var(pc2) >= np.var(pc3)
        
        # All should be positive
        assert np.var(pc1) > 0
        assert np.var(pc2) > 0
        assert np.var(pc3) > 0


class TestPCAProperties:
    """Property-based tests for PCA invariants."""

    # Strategy to generate valid DataFrames for PCA
    valid_df_strategy = data_frames(
        columns=[
            column('f1', elements=st.floats(min_value=-100, max_value=100, allow_nan=False, allow_infinity=False)),
            column('f2', elements=st.floats(min_value=-100, max_value=100, allow_nan=False, allow_infinity=False)),
            column('f3', elements=st.floats(min_value=-100, max_value=100, allow_nan=False, allow_infinity=False)),
            column('f4', elements=st.floats(min_value=-100, max_value=100, allow_nan=False, allow_infinity=False)),
            column('class', elements=st.integers(min_value=0, max_value=2))
        ],
        index=range_indexes(min_size=20)
    ).filter(lambda df: df['class'].nunique() >= 2)  # Need at least 2 classes for meaningful testing

    @settings(deadline=None, max_examples=30)
    @given(df=valid_df_strategy)
    def test_property_invariant_to_scaling(self, df):
        """
        Property: Scaling all input features by a constant factor should 
        not change the relative directions of principal components.
        """
        # Run on original data
        res1 = next(pca_mod.run_pca(df.copy(), num_eigenvector=2, target_col='class'))
        
        # Run on scaled data
        df_scaled = df.copy()
        df_scaled[['f1', 'f2', 'f3', 'f4']] *= 10.0
        res2 = next(pca_mod.run_pca(df_scaled, num_eigenvector=2, target_col='class'))
        
        # The absolute correlation between PCs should be 1.0 (sign may flip)
        corr_pc1 = np.abs(np.corrcoef(res1['PC1'], res2['PC1'])[0, 1])
        corr_pc2 = np.abs(np.corrcoef(res1['PC2'], res2['PC2'])[0, 1])
        
        assert corr_pc1 > 0.999
        assert corr_pc2 > 0.999

    @settings(deadline=None, max_examples=30)
    @given(df=valid_df_strategy)
    def test_property_output_dimensions(self, df):
        """
        Property: The output should always contain exactly num_eigenvector PC columns
        regardless of the number of input features.
        """
        n_vecs = 3
        result = next(pca_mod.run_pca(df, num_eigenvector=n_vecs, target_col='class'))
        
        pc_cols = [c for c in result.columns if c.startswith('PC')]
        assert len(pc_cols) == n_vecs
        assert len(result) == len(df)

    @settings(deadline=None, max_examples=25)
    @given(df=valid_df_strategy)
    def test_property_variance_ordering(self, df):
        """
        Property: Principal components should be ordered by decreasing variance.
        """
        n_components = 3
        result = next(pca_mod.run_pca(df, num_eigenvector=n_components, target_col='class'))
        
        pc_cols = [c for c in result.columns if c.startswith('PC')]
        variances = [result[col].var() for col in pc_cols]
        
        # Variances should be in non-increasing order
        for i in range(len(variances) - 1):
            assert variances[i] >= variances[i + 1]

    @settings(deadline=None, max_examples=25)
    @given(df=valid_df_strategy)
    def test_property_orthogonality(self, df):
        """
        Property: Principal components should be orthogonal to each other.
        """
        try:
            n_components = 3
            result = next(pca_mod.run_pca(df, num_eigenvector=n_components, target_col='class'))
            
            pc_cols = [c for c in result.columns if c.startswith('PC')]
            
            # Check pairwise orthogonality
            for i in range(len(pc_cols)):
                for j in range(i + 1, len(pc_cols)):
                    correlation = np.abs(np.corrcoef(result[pc_cols[i]], result[pc_cols[j]])[0, 1])
                    # Handle NaN correlations
                    if np.isnan(correlation):
                        continue
                    assert correlation < 0.1  # Should be nearly orthogonal
        except (ValueError, np.linalg.LinAlgError):
            # Skip if PCA fails on pathological data
            pass

    @settings(deadline=None, max_examples=20)
    @given(df=valid_df_strategy)
    def test_property_centering_independence(self, df):
        """
        Property: Adding a constant to all features (translation) should not change PC directions.
        """
        try:
            # Run on original data
            res1 = next(pca_mod.run_pca(df.copy(), num_eigenvector=2, target_col='class'))
            
            # Translate data
            df_translated = df.copy()
            df_translated[['f1', 'f2', 'f3', 'f4']] += 50.0
            res2 = next(pca_mod.run_pca(df_translated, num_eigenvector=2, target_col='class'))
            
            # Correlations should be 1.0 (sign may flip)
            corr_pc1 = np.abs(np.corrcoef(res1['PC1'], res2['PC1'])[0, 1])
            corr_pc2 = np.abs(np.corrcoef(res1['PC2'], res2['PC2'])[0, 1])
            
            # Handle NaN correlations
            if not np.isnan(corr_pc1):
                assert corr_pc1 > 0.999
            if not np.isnan(corr_pc2):
                assert corr_pc2 > 0.999
        except (ValueError, np.linalg.LinAlgError):
            # Skip if PCA fails on pathological data
            pass

    @settings(deadline=None, max_examples=20)
    @given(df=valid_df_strategy)
    def test_property_class_preservation(self, df):
        """
        Property: The class column should be preserved exactly in the output.
        """
        result = next(pca_mod.run_pca(df, num_eigenvector=2, target_col='class'))
        
        # Class column should be identical
        pd.testing.assert_series_equal(result['class'], df['class'])
        
        # Number of rows should be preserved
        assert len(result) == len(df)

    @settings(deadline=None, max_examples=15)
    @given(df=valid_df_strategy)
    def test_property_feature_permutation_invariance(self, df):
        """
        Property: Permuting feature columns should not change the set of PC values
        (though they may appear in different order).
        """
        # Original order
        res1 = next(pca_mod.run_pca(df.copy(), num_eigenvector=2, target_col='class'))
        
        # Permute feature columns (keep class at end)
        feature_cols = [c for c in df.columns if c != 'class']
        df_permuted = df[feature_cols[::-1] + ['class']].copy()
        res2 = next(pca_mod.run_pca(df_permuted, num_eigenvector=2, target_col='class'))
        
        # The multiset of PC values should be the same (up to sign and order)
        pc_values_1 = set(np.abs(np.concatenate([res1['PC1'].values, res1['PC2'].values])))
        pc_values_2 = set(np.abs(np.concatenate([res2['PC1'].values, res2['PC2'].values])))
        
        # Most values should be very close (allowing for numerical precision)
        close_values = sum(1 for v1 in pc_values_1 for v2 in pc_values_2 if abs(v1 - v2) < 1e-10)
        assert close_values >= min(len(pc_values_1), len(pc_values_2)) * 0.9

    @settings(deadline=None, max_examples=15)
    @given(df=valid_df_strategy)
    def test_property_variance_explained_monotonicity(self, df):
        """
        Property: Cumulative variance explained should be monotonic increasing.
        """
        n_features = len([c for c in df.columns if c != 'class'])
        result = next(pca_mod.run_pca(df, num_eigenvector=n_features, target_col='class'))
        
        pc_cols = [c for c in result.columns if c.startswith('PC')]
        variances = [result[col].var() for col in pc_cols]
        
        # Cumulative variance should be monotonic
        cumulative = np.cumsum(variances)
        for i in range(len(cumulative) - 1):
            assert cumulative[i] <= cumulative[i + 1]

    @settings(deadline=None, max_examples=10)
    @given(df=valid_df_strategy)
    def test_property_deterministic_behavior(self, df):
        """
        Property: PCA should be deterministic - same input should produce same output.
        """
        res1 = next(pca_mod.run_pca(df.copy(), num_eigenvector=2, target_col='class'))
        res2 = next(pca_mod.run_pca(df.copy(), num_eigenvector=2, target_col='class'))
        
        # Results should be identical (PCA is deterministic)
        pd.testing.assert_frame_equal(res1.sort_index(), res2.sort_index())


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
