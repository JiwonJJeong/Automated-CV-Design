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
module_path = os.path.join(BASE_DIR, 'lda', '4_dimensionality_reduction', 'ZHLDA.py')
spec = importlib.util.spec_from_file_location("zhlda_mod", module_path)
zhlda_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(zhlda_mod)
import dimensionality_reduction_utils as utils

# Reference files
REF_FILE = os.path.join(BASE_DIR, "tests", "4_dimensionality_reduction", "ZHLDA.csv")


class TestZHLDA:
    """Comprehensive test suite for Zero-order Heteroscedastic LDA."""
    
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
    
    def test_zhlda_basic_functionality(self, sample_dataframe):
        """Test basic ZHLDA functionality."""
        result_iter = zhlda_mod.run_zhlda(sample_dataframe, num_eigenvector=2, target_col='class')
        result_df = next(result_iter)
        
        assert isinstance(result_df, pd.DataFrame)
        assert 'class' in result_df.columns
        assert 'LD1' in result_df.columns
        assert 'LD2' in result_df.columns
        assert len(result_df) == len(sample_dataframe)
        assert len(result_df.columns) == 3  # LD1, LD2, class
    
    def test_zhlda_single_eigenvector(self, sample_dataframe):
        """Test ZHLDA with single eigenvector."""
        result_iter = zhlda_mod.run_zhlda(sample_dataframe, num_eigenvector=1, target_col='class')
        result_df = next(result_iter)
        
        assert isinstance(result_df, pd.DataFrame)
        assert 'class' in result_df.columns
        assert 'LD1' in result_df.columns
        assert 'LD2' not in result_df.columns
        assert len(result_df.columns) == 2  # LD1, class
    
    def test_zhlda_three_eigenvectors(self, sample_dataframe):
        """Test ZHLDA with three eigenvectors (should be capped at 2 for 3 classes)."""
        result_iter = zhlda_mod.run_zhlda(sample_dataframe, num_eigenvector=3, target_col='class')
        result_df = next(result_iter)
        
        assert isinstance(result_df, pd.DataFrame)
        assert 'class' in result_df.columns
        assert 'LD1' in result_df.columns
        assert 'LD2' in result_df.columns
        # LD3 should NOT be present for 3 classes (max is classes-1 = 2)
        assert 'LD3' not in result_df.columns
        assert len(result_df.columns) == 3  # LD1, LD2, class
    
    def test_zhlda_with_two_classes(self):
        """Test ZHLDA with binary classification."""
        np.random.seed(42)
        binary_df = pd.DataFrame({
            'feature_0': np.concatenate([np.random.normal(0, 1, 50), np.random.normal(3, 1, 50)]),
            'feature_1': np.concatenate([np.random.normal(0, 1, 50), np.random.normal(3, 1, 50)]),
            'class': [1] * 50 + [2] * 50
        })
        
        result_iter = zhlda_mod.run_zhlda(binary_df, num_eigenvector=1, target_col='class')
        result_df = next(result_iter)
        
        assert isinstance(result_df, pd.DataFrame)
        assert 'class' in result_df.columns
        assert 'LD1' in result_df.columns
        assert len(result_df) == len(binary_df)
    
    def test_zhlda_with_small_dataset(self):
        """Test ZHLDA with very small datasets."""
        tiny_df = pd.DataFrame({
            'feature_0': [1, 2, 3, 4, 5, 6],
            'feature_1': [2, 3, 4, 5, 6, 7],
            'class': [1, 1, 2, 2, 3, 3]
        })
        
        try:
            result_iter = zhlda_mod.run_zhlda(tiny_df, num_eigenvector=2, target_col='class')
            result_df = next(result_iter)
            
            assert isinstance(result_df, pd.DataFrame)
            assert 'class' in result_df.columns
            assert len(result_df) == len(tiny_df)
        except np.linalg.LinAlgError:
            # Small datasets can cause singular matrices - this is acceptable
            pytest.skip("Singular matrix encountered with tiny dataset")
    
    def test_zhlda_with_no_discriminative_features(self):
        """Test ZHLDA with random noise features."""
        np.random.seed(42)
        no_signal_df = pd.DataFrame({
            'feature_0': np.random.normal(0, 1, 90),
            'feature_1': np.random.normal(0, 1, 90),
            'feature_2': np.random.normal(0, 1, 90),
            'class': [1] * 30 + [2] * 30 + [3] * 30
        })
        
        result_iter = zhlda_mod.run_zhlda(no_signal_df, num_eigenvector=2, target_col='class')
        result_df = next(result_iter)
        
        assert isinstance(result_df, pd.DataFrame)
        assert 'class' in result_df.columns
        assert len(result_df) == len(no_signal_df)
    
    def test_zhlda_zero_order_properties(self, sample_dataframe):
        """Test zero-order specific properties of ZHLDA."""
        result_iter = zhlda_mod.run_zhlda(sample_dataframe, num_eigenvector=2, target_col='class')
        result_df = next(result_iter)
        
        # Zero-order methods should be computationally efficient
        # Test that it handles larger datasets reasonably well
        large_df = pd.DataFrame({
            f'feature_{i}': np.random.randn(1000) for i in range(20)
        })
        large_df['class'] = np.random.choice([1, 2, 3], 1000)
        
        result_iter = zhlda_mod.run_zhlda(large_df, num_eigenvector=2, target_col='class')
        result_df = next(result_iter)
        
        assert isinstance(result_df, pd.DataFrame)
        assert 'class' in result_df.columns
        assert len(result_df) == len(large_df)
    
    def test_zhlda_approximation_quality(self, sample_dataframe):
        """Test that ZHLDA provides reasonable approximations."""
        result_iter = zhlda_mod.run_zhlda(sample_dataframe, num_eigenvector=2, target_col='class')
        result_df = next(result_iter)
        
        # Calculate class separation in LD space
        classes = result_df['class'].unique()
        class_means = []
        
        for cls in classes:
            class_data = result_df[result_df['class'] == cls][['LD1', 'LD2']]
            class_means.append(class_data.mean().values)
        
        # Check that class means are reasonably separated
        class_means = np.array(class_means)
        
        # Calculate pairwise distances between class means
        distances = []
        for i in range(len(class_means)):
            for j in range(i + 1, len(class_means)):
                dist = np.linalg.norm(class_means[i] - class_means[j])
                distances.append(dist)
        
        # At least some separation should exist
        assert np.mean(distances) > 0.3  # Lower threshold for zero-order approximation
    
    def test_error_handling_invalid_target(self, sample_dataframe):
        """Test error handling with invalid target column."""
        with pytest.raises(ValueError):
            next(zhlda_mod.run_zhlda(sample_dataframe, num_eigenvector=2, target_col='invalid_target'))
    
    def test_error_handling_insufficient_classes(self):
        """Test error handling with insufficient classes."""
        single_class_df = pd.DataFrame({
            'feature_0': [1, 2, 3, 4],
            'feature_1': [2, 3, 4, 5],
            'class': [1, 1, 1, 1]
        })
        
        # Should handle single class gracefully or raise appropriate error
        try:
            result_iter = zhlda_mod.run_zhlda(single_class_df, num_eigenvector=2, target_col='class')
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
            result_iter = zhlda_mod.run_zhlda(empty_df, num_eigenvector=2, target_col='class')
            result_df = next(result_iter)
            assert isinstance(result_df, pd.DataFrame)
        except (ValueError, IndexError):
            # These errors are acceptable for empty data
            pass
    
    def test_integration_with_real_data(self):
        """Integration test using real data."""
        try:
            df = utils.get_mpso_data()
            df = utils.assign_classes(df, start_label=1)
            
            result_iter = zhlda_mod.run_zhlda(df, num_eigenvector=2, target_col='class')
            result_df = next(result_iter)
            
            assert isinstance(result_df, pd.DataFrame)
            assert 'class' in result_df.columns
            assert 'LD1' in result_df.columns
            assert 'LD2' in result_df.columns
            assert len(result_df) == len(df)
            
        except FileNotFoundError:
            pytest.skip("Real test data not available")
    
    def test_reference_output_comparison(self):
        """Test against reference output if available."""
        if not os.path.exists(REF_FILE):
            pytest.skip("Reference file not available")
        
        try:
            df = utils.get_mpso_data()
            df = utils.assign_classes(df, start_label=1)
            
            result_iter = zhlda_mod.run_zhlda(df, num_eigenvector=2, target_col='class')
            result_df = next(result_iter)
            
            ref_df = pd.read_csv(REF_FILE)
            
            # Check shape
            assert result_df.shape == ref_df.shape, f"Shape mismatch: {result_df.shape} vs {ref_df.shape}"
            
            # Check class column exactly
            pd.testing.assert_series_equal(result_df['class'], ref_df['class'])
            
            # Check LD values (with sign handling)
            for col in ['LD1', 'LD2']:
                diff_pos = np.abs(result_df[col] - ref_df[col]).mean()
                diff_neg = np.abs(result_df[col] + ref_df[col]).mean()
                
                assert min(diff_pos, diff_neg) < 1e-3, f"Values in {col} do not match reference"
                
        except FileNotFoundError:
            pytest.skip("Real test data not available")
    
    def test_computational_efficiency(self, sample_dataframe):
        """Test that ZHLDA is computationally efficient."""
        import time
        
        # Test with a moderately large dataset
        large_df = pd.DataFrame({
            f'feature_{i}': np.random.randn(500) for i in range(15)
        })
        large_df['class'] = np.random.choice([1, 2, 3], 500)
        
        start_time = time.time()
        result_iter = zhlda_mod.run_zhlda(large_df, num_eigenvector=2, target_col='class')
        result_df = next(result_iter)
        end_time = time.time()
        
        # Should complete within reasonable time (adjust threshold as needed)
        assert end_time - start_time < 10.0  # 10 seconds max
        assert isinstance(result_df, pd.DataFrame)
        assert 'class' in result_df.columns
    
    def test_numerical_stability(self, sample_dataframe):
        """Test numerical stability with extreme values."""
        # Add some extreme values to test numerical stability
        extreme_df = sample_dataframe.copy()
        extreme_df.loc[0, 'feature_0'] = 1e6
        extreme_df.loc[1, 'feature_0'] = -1e6
        
        try:
            result_iter = zhlda_mod.run_zhlda(extreme_df, num_eigenvector=2, target_col='class')
            result_df = next(result_iter)
            
            assert isinstance(result_df, pd.DataFrame)
            assert 'class' in result_df.columns
            assert not np.any(np.isnan(result_df[['LD1', 'LD2']].values))
            
        except (ValueError, np.linalg.LinAlgError):
            # Numerical instability is acceptable for extreme values
            pass
    
    def test_reproducibility(self, sample_dataframe):
        """Test that ZHLDA produces reproducible results."""
        np.random.seed(42)
        
        result_iter1 = zhlda_mod.run_zhlda(sample_dataframe, num_eigenvector=2, target_col='class')
        result_df1 = next(result_iter1)
        
        np.random.seed(42)
        
        result_iter2 = zhlda_mod.run_zhlda(sample_dataframe, num_eigenvector=2, target_col='class')
        result_df2 = next(result_iter2)
        
        # Results should be identical (ZHLDA should be deterministic)
        pd.testing.assert_frame_equal(result_df1.sort_index(), result_df2.sort_index())
    
    def test_different_class_labels(self):
        """Test ZHLDA with different class label schemes."""
        np.random.seed(42)
        
        # Test with 0, 1, 2 labels
        df_012 = pd.DataFrame({
            'feature_0': np.concatenate([np.random.normal(0, 1, 30), np.random.normal(3, 1, 30), np.random.normal(6, 1, 30)]),
            'class': [0, 1, 2] * 30
        })
        
        result_iter = zhlda_mod.run_zhlda(df_012, num_eigenvector=2, target_col='class')
        result_df = next(result_iter)
        
        assert isinstance(result_df, pd.DataFrame)
        assert 'class' in result_df.columns
        assert len(result_df) == len(df_012)
    
    def test_zero_order_approximation_behavior(self, sample_dataframe):
        """Test specific behaviors of zero-order approximation."""
        # Zero-order methods should be robust to noise
        noisy_df = sample_dataframe.copy()
        
        # Add significant noise
        for col in noisy_df.columns:
            if col != 'class':
                noisy_df[col] += np.random.normal(0, 0.5, len(noisy_df))
        
        result_iter = zhlda_mod.run_zhlda(noisy_df, num_eigenvector=2, target_col='class')
        result_df = next(result_iter)
        
        assert isinstance(result_df, pd.DataFrame)
        assert 'class' in result_df.columns
        assert not np.any(np.isnan(result_df[['LD1', 'LD2']].values))
    
    def test_eigenvector_properties(self, sample_dataframe):
        """Test properties of the computed eigenvectors."""
        result_iter = zhlda_mod.run_zhlda(sample_dataframe, num_eigenvector=2, target_col='class')
        result_df = next(result_iter)
        
        # Extract LD components
        ld1 = result_df['LD1'].values
        ld2 = result_df['LD2'].values
        
        # Check that components have non-zero variance
        assert np.var(ld1) > 1e-6
        assert np.var(ld2) > 1e-6
        
        # Check that components provide some discrimination
        class_correlations = []
        for cls in result_df['class'].unique():
            class_mask = result_df['class'] == cls
            class_correlations.append(np.corrcoef(ld1[class_mask], ld2[class_mask])[0, 1] if len(ld1[class_mask]) > 1 else 0)
        
        # At least some classes should have different correlation patterns
        assert len(set(round(c, 2) for c in class_correlations)) > 1


class TestZHLDAProperties:
    """Property-based tests for ZHLDA invariants."""

    # Strategy to generate valid DataFrames for LDA
    valid_df_strategy = data_frames(
        columns=[
            column('f1', elements=st.floats(min_value=-1e3, max_value=1e3, allow_nan=False, allow_infinity=False)),
            column('f2', elements=st.floats(min_value=-1e3, max_value=1e3, allow_nan=False, allow_infinity=False)),
            column('f3', elements=st.floats(min_value=-1e3, max_value=1e3, allow_nan=False, allow_infinity=False)),
            column('class', elements=st.integers(min_value=1, max_value=3))
        ],
        index=range_indexes(min_size=20)
    ).filter(lambda df: df['class'].nunique() >= 2) # LDA requires at least 2 classes

    @settings(deadline=None, max_examples=25)
    @given(df=valid_df_strategy)
    def test_property_invariant_to_scaling(self, df):
        """
        Property: Scaling all input features by a constant factor should 
        not change the relative projections (direction of LDs).
        """
        try:
            # Run on original data
            res1 = next(zhlda_mod.run_zhlda(df.copy(), num_eigenvector=1, target_col='class'))
            
            # Run on scaled data
            df_scaled = df.copy()
            df_scaled[['f1', 'f2', 'f3']] *= 10.0
            res2 = next(zhlda_mod.run_zhlda(df_scaled, num_eigenvector=1, target_col='class'))
            
            # The absolute correlation between projections should be near 1.0
            correlation = np.abs(np.corrcoef(res1['LD1'], res2['LD1'])[0, 1])
            assert correlation > 0.95
        except (np.linalg.LinAlgError, ValueError):
            # Gradient descent may fail on singular matrices generated by chance
            pass

    @settings(deadline=None, max_examples=25)
    @given(df=valid_df_strategy)
    def test_property_output_dimensions(self, df):
        """
        Property: The output should contain the correct number of LD columns
        capped by the theoretical maximum (classes - 1).
        """
        n_vecs = 2
        result = next(zhlda_mod.run_zhlda(df, num_eigenvector=n_vecs, target_col='class'))
        
        ld_cols = [c for c in result.columns if c.startswith('LD')]
        max_possible = min(len(df.columns) - 1, df['class'].nunique() - 1)
        expected_lds = min(n_vecs, max_possible)
        
        assert len(ld_cols) == expected_lds
        assert len(result) == len(df)

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
            result = next(zhlda_mod.run_zhlda(df, num_eigenvector=10, target_col='class'))
            
            ld_cols = [c for c in result.columns if c.startswith('LD')]
            assert len(ld_cols) <= max_ld
        except np.linalg.LinAlgError:
            # Skip on singular matrices
            pass

    @given(st.data())
    def test_property_global_mean_centering_independence(self, data):
        """
        Property: Adding a constant bias to a feature (translation) 
        should not change the LD projection values for zero-mean LDA.
        """
        df = data.draw(self.valid_df_strategy)
        
        try:
            res1 = next(zhlda_mod.run_zhlda(df.copy(), num_eigenvector=1, target_col='class'))
            
            # Translate data
            df_translated = df.copy()
            df_translated[['f1', 'f2', 'f3']] += 100.0
            res2 = next(zhlda_mod.run_zhlda(df_translated, num_eigenvector=1, target_col='class'))
            
            # Use a tolerance for floating point drift in gradient descent
            np.testing.assert_allclose(res1['LD1'].values, res2['LD1'].values, atol=1e-2)
        except (np.linalg.LinAlgError, ValueError):
            pass

    @settings(deadline=None, max_examples=15)
    @given(df=valid_df_strategy)
    def test_property_class_preservation(self, df):
        """
        Property: The class column should be preserved exactly in the output.
        """
        result = next(zhlda_mod.run_zhlda(df, num_eigenvector=1, target_col='class'))
        
        # Class column should be identical
        pd.testing.assert_series_equal(result['class'], df['class'])
        
        # Number of rows should be preserved
        assert len(result) == len(df)

    @settings(deadline=None, max_examples=15)
    @given(df=valid_df_strategy)
    def test_property_orthogonality_maintenance(self, df):
        """
        Property: Multiple LD components should remain approximately orthogonal.
        """
        if df['class'].nunique() < 3:
            pytest.skip("Need at least 3 classes for 2 LD components")
        
        try:
            result = next(zhlda_mod.run_zhlda(df, num_eigenvector=2, target_col='class'))
            
            ld1 = result['LD1'].values
            ld2 = result['LD2'].values
            
            # Check orthogonality (correlation should be near 0)
            correlation = np.abs(np.corrcoef(ld1, ld2)[0, 1])
            assert correlation < 0.3  # Should be approximately orthogonal
        except (np.linalg.LinAlgError, ValueError):
            pass

    @settings(deadline=None, max_examples=10)
    @given(df=valid_df_strategy)
    def test_property_convergence_behavior(self, df):
        """
        Property: ZHLDA should converge to a stable solution.
        """
        try:
            # Run with reduced iterations for testing
            result = next(zhlda_mod.run_zhlda(
                df, num_eigenvector=1, target_col='class', 
                learning_rate=0.001, num_iteration=100
            ))
            
            # Should produce valid output
            assert 'LD1' in result.columns
            assert 'class' in result.columns
            assert len(result) == len(df)
            
            # LD values should not be all zeros or NaN
            ld_values = result['LD1'].values
            assert not np.allclose(ld_values, 0)
            assert not np.any(np.isnan(ld_values))
        except (np.linalg.LinAlgError, ValueError):
            pass


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
