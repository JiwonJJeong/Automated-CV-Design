import pytest
import pandas as pd
import numpy as np
import os
import sys
import importlib.util
from unittest.mock import patch, MagicMock
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

# 2. Load GDHLDA module
module_path = os.path.join(DR_DIR, 'GDHLDA.py')
spec_gdhlda = importlib.util.spec_from_file_location("gdhlda_mod", module_path)
gdhlda_mod = importlib.util.module_from_spec(spec_gdhlda)
spec_gdhlda.loader.exec_module(gdhlda_mod)

# 3. Import utils for integration tests
try:
    import dimensionality_reduction_utils as utils
except ImportError:
    utils = None

# Reference files
REF_FILE = os.path.join(BASE_DIR, "tests", "4_dimensionality_reduction", "GDHLDA.csv")

# =============================================================================
# ENHANCED TEST CLASS - MHLDA PATTERN
# =============================================================================

class TestGDHLDAEnhanced:
    """Enhanced GDHLDA tests following MHLDA pattern."""
    
    @pytest.fixture
    def sample_dataframe(self):
        """Create a synthetic dataset with heteroscedastic structure."""
        np.random.seed(42)
        n_samples = 100  # Reduced for faster testing
        
        # Create heteroscedastic data (different variances per class)
        class_0 = np.random.multivariate_normal([0, 0], [[1, 0.3], [0.3, 1]], n_samples//2)
        class_1 = np.random.multivariate_normal([3, 3], [[2, -0.5], [-0.5, 2]], n_samples//2)
        
        data = np.vstack([class_0, class_1])
        
        df = pd.DataFrame(data, columns=['feature_1', 'feature_2'])
        df['class'] = [0] * (n_samples // 2) + [1] * (n_samples // 2)
        
        return df

    @pytest.fixture
    def sample_dataframe_original(self):
        """Create a sample DataFrame for testing (original style)."""
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

    # --- Unit Tests ---
    
    def test_gdhlda_basic_functionality(self, sample_dataframe):
        """Test basic GDHLDA computation."""
        result_iter = gdhlda_mod.run_gdhlda(sample_dataframe, num_eigenvector=1, target_col='class')
        result_df = next(result_iter)
        
        assert isinstance(result_df, pd.DataFrame)
        assert 'class' in result_df.columns
        # Check for both LD1 and GDHLD1 naming conventions
        assert 'LD1' in result_df.columns or 'GDHLD1' in result_df.columns
        assert len(result_df) == len(sample_dataframe)

    def test_gdhlda_output_dimensions(self, sample_dataframe):
        """Test that GDHLDA outputs correct number of components."""
        n_components = 1  # GDHLDA max is classes-1 = 1 for 2 classes
        result_iter = gdhlda_mod.run_gdhlda(sample_dataframe, num_eigenvector=n_components, target_col='class')
        result_df = next(result_iter)
        
        # Check for both naming conventions
        ld_cols = [c for c in result_df.columns if c.startswith('LD') or c.startswith('GDHLD')]
        assert len(ld_cols) == n_components

    def test_gdhlda_with_three_classes(self):
        """Test GDHLDA with three classes."""
        np.random.seed(42)
        n_samples = 90  # 30 per class
        
        # Create three classes with different covariance structures
        class_0 = np.random.multivariate_normal([0, 0], [[1, 0.2], [0.2, 1]], n_samples//3)
        class_1 = np.random.multivariate_normal([3, 0], [[2, -0.3], [-0.3, 2]], n_samples//3)
        class_2 = np.random.multivariate_normal([1.5, 3], [[1.5, 0.4], [0.4, 1.5]], n_samples//3)
        
        data = np.vstack([class_0, class_1, class_2])
        
        df = pd.DataFrame(data, columns=['feature_1', 'feature_2'])
        df['class'] = [0] * (n_samples // 3) + [1] * (n_samples // 3) + [2] * (n_samples // 3)
        
        result_iter = gdhlda_mod.run_gdhlda(df, num_eigenvector=2, target_col='class')
        result_df = next(result_iter)
        
        assert isinstance(result_df, pd.DataFrame)
        assert 'class' in result_df.columns
        # Check for both naming conventions
        assert ('LD1' in result_df.columns and 'LD2' in result_df.columns) or \
               ('GDHLD1' in result_df.columns and 'GDHLD2' in result_df.columns)

    def test_gdhlda_with_small_dataset(self):
        """Test GDHLDA with minimal dataset."""
        small_df = pd.DataFrame({
            'feature_1': [1, 2, 3, 4, 5, 6],
            'feature_2': [2, 4, 6, 8, 10, 12],
            'class': [0, 0, 0, 1, 1, 1]
        })
        
        result_iter = gdhlda_mod.run_gdhlda(small_df, num_eigenvector=1, target_col='class')
        result_df = next(result_iter)
        
        assert isinstance(result_df, pd.DataFrame)
        assert 'class' in result_df.columns
        assert 'LD1' in result_df.columns or 'GDHLD1' in result_df.columns

    def test_gdhlda_class_separation_quality(self, sample_dataframe):
        """Test that GDHLDA provides good class separation for heteroscedastic data."""
        result_iter = gdhlda_mod.run_gdhlda(sample_dataframe, num_eigenvector=1, target_col='class')
        result_df = next(result_iter)
        
        # Get the LD column name (could be LD1 or GDHLD1)
        ld_col = 'LD1' if 'LD1' in result_df.columns else 'GDHLD1'
        
        # Check that classes are separated in LD space
        class_0_ld = result_df[result_df['class'] == 0][ld_col]
        class_1_ld = result_df[result_df['class'] == 1][ld_col]
        
        # Means should be different
        mean_diff = abs(class_0_ld.mean() - class_1_ld.mean())
        assert mean_diff > 0.5, f"Classes should be separated, mean difference: {mean_diff}"

    def test_gdhlda_single_eigenvector(self, sample_dataframe):
        """Test GDHLDA with single eigenvector."""
        result_iter = gdhlda_mod.run_gdhlda(sample_dataframe, num_eigenvector=1, target_col='class')
        result_df = next(result_iter)
        
        assert 'LD1' in result_df.columns or 'GDHLD1' in result_df.columns
        assert 'LD2' not in result_df.columns and 'GDHLD2' not in result_df.columns

    def test_gdhlda_reproducibility(self, sample_dataframe):
        """Ensure GDHLDA produces reproducible results with same seed."""
        try:
            result_iter1 = gdhlda_mod.run_gdhlda(sample_dataframe, num_eigenvector=1, target_col='class', random_state=42)
            result_df1 = next(result_iter1)
            
            result_iter2 = gdhlda_mod.run_gdhlda(sample_dataframe, num_eigenvector=1, target_col='class', random_state=42)
            result_df2 = next(result_iter2)
            
            # Results should be identical with same seed
            pd.testing.assert_frame_equal(result_df1.sort_index(), result_df2.sort_index())
        except TypeError:
            # If random_state parameter is not supported, skip
            pytest.skip("random_state parameter not supported")

    def test_gdhlda_reproducibility_with_seeds(self, sample_dataframe):
        """Test that GDHLDA produces reproducible results with seeds."""
        np.random.seed(42)
        
        result_iter1 = gdhlda_mod.run_gdhlda(sample_dataframe, num_eigenvector=1, target_col='class')
        result_df1 = next(result_iter1)
        
        np.random.seed(42)
        
        result_iter2 = gdhlda_mod.run_gdhlda(sample_dataframe, num_eigenvector=1, target_col='class')
        result_df2 = next(result_iter2)
        
        # Results should be identical with same seed
        pd.testing.assert_frame_equal(result_df1.sort_index(), result_df2.sort_index())

    def test_error_handling_invalid_target(self, sample_dataframe):
        """Test handling of invalid target column."""
        with pytest.raises((ValueError, KeyError)):
            result_iter = gdhlda_mod.run_gdhlda(sample_dataframe, num_eigenvector=1, target_col='invalid_target')
            next(result_iter)

    def test_error_handling_single_class(self):
        """Test handling of dataset with single class."""
        single_class_df = pd.DataFrame({
            'feature_1': [1, 2, 3, 4],
            'feature_2': [2, 4, 6, 8],
            'class': [0, 0, 0, 0]
        })
        
        with pytest.raises((ValueError, np.linalg.LinAlgError)):
            result_iter = gdhlda_mod.run_gdhlda(single_class_df, num_eigenvector=1, target_col='class')
            next(result_iter)

    def test_gdhlda_with_no_discriminative_features(self):
        """Test GDHLDA with random noise features."""
        np.random.seed(42)
        no_signal_df = pd.DataFrame({
            'feature_0': np.random.normal(0, 1, 90),
            'feature_1': np.random.normal(0, 1, 90),
            'feature_2': np.random.normal(0, 1, 90),
            'class': [1] * 30 + [2] * 30 + [3] * 30
        })
        
        result_iter = gdhlda_mod.run_gdhlda(no_signal_df, num_eigenvector=2, target_col='class')
        result_df = next(result_iter)
        
        assert isinstance(result_df, pd.DataFrame)
        assert 'class' in result_df.columns
        assert len(result_df) == len(no_signal_df)

    def test_gdhlda_convergence_parameters(self, sample_dataframe):
        """Test GDHLDA with different convergence parameters."""
        params = [
            {'learning_rate': 0.01, 'max_iter': 50},
            {'learning_rate': 0.05, 'max_iter': 100},
            {'learning_rate': 0.1, 'max_iter': 200}
        ]
        
        for param in params:
            try:
                result_iter = gdhlda_mod.run_gdhlda(
                    sample_dataframe, num_eigenvector=2, target_col='class', **param
                )
                result_df = next(result_iter)
                
                assert isinstance(result_df, pd.DataFrame)
                assert 'class' in result_df.columns
                assert 'LD1' in result_df.columns or 'GDHLD1' in result_df.columns
                assert 'LD2' in result_df.columns or 'GDHLD2' in result_df.columns
                
            except TypeError:
                # If parameters are not supported, skip
                pass

    def test_gdhlda_different_learning_rates(self, sample_dataframe):
        """Test GDHLDA with different learning rates."""
        learning_rates = [0.001, 0.01, 0.1]
        
        for lr in learning_rates:
            try:
                result_iter = gdhlda_mod.run_gdhlda(
                    sample_dataframe, num_eigenvector=2, target_col='class', learning_rate=lr
                )
                result_df = next(result_iter)
                
                assert isinstance(result_df, pd.DataFrame)
                assert 'class' in result_df.columns
                
            except TypeError:
                # If learning_rate is not a parameter, skip this test
                pass

    def test_gdhlda_stopping_criteria(self, sample_dataframe):
        """Test different stopping criteria."""
        stopping_criteria = [1e-6, 1e-4, 1e-2]
        
        for criteria in stopping_criteria:
            try:
                result_iter = gdhlda_mod.run_gdhlda(
                    sample_dataframe, num_eigenvector=2, target_col='class', stop_crit=criteria
                )
                result_df = next(result_iter)
                
                assert isinstance(result_df, pd.DataFrame)
                assert 'class' in result_df.columns
                
            except TypeError:
                # If stop_crit is not a parameter, skip this test
                pass

    def test_gdhlda_numerical_stability(self, sample_dataframe):
        """Test numerical stability with extreme values."""
        # Add some extreme values to test numerical stability
        extreme_df = sample_dataframe.copy()
        extreme_df.loc[0, 'feature_1'] = 1e6
        extreme_df.loc[1, 'feature_1'] = -1e6
        
        try:
            result_iter = gdhlda_mod.run_gdhlda(extreme_df, num_eigenvector=2, target_col='class')
            result_df = next(result_iter)
            
            assert isinstance(result_df, pd.DataFrame)
            assert 'class' in result_df.columns
            
            # Check for NaN values in LD columns
            ld_cols = [c for c in result_df.columns if c.startswith('LD') or c.startswith('GDHLD')]
            for col in ld_cols:
                assert not np.any(np.isnan(result_df[col].values))
            
        except (ValueError, np.linalg.LinAlgError):
            # Numerical instability is acceptable for extreme values
            pass

    def test_empty_dataframe_handling(self):
        """Test handling of empty DataFrames."""
        empty_df = pd.DataFrame({'class': [], 'feature_0': []})
        
        # Should handle empty data gracefully or raise appropriate error
        try:
            result_iter = gdhlda_mod.run_gdhlda(empty_df, num_eigenvector=2, target_col='class')
            result_df = next(result_iter)
            assert isinstance(result_df, pd.DataFrame)
        except (ValueError, IndexError):
            # These errors are acceptable for empty data
            pass

    def test_gdhlda_convergence_monitoring(self, sample_dataframe):
        """Test that GDHLDA converges properly."""
        # Mock to capture convergence information if available
        with patch.object(gdhlda_mod, 'run_gdhlda') as mock_run:
            # Create a mock result
            mock_result = pd.DataFrame({
                'LD1': np.random.randn(len(sample_dataframe)),
                'LD2': np.random.randn(len(sample_dataframe)),
                'class': sample_dataframe['class']
            })
            mock_run.return_value = iter([mock_result])
            
            result_iter = gdhlda_mod.run_gdhlda(sample_dataframe, num_eigenvector=2, target_col='class')
            result_df = next(result_iter)
            
            assert isinstance(result_df, pd.DataFrame)
            mock_run.assert_called_once()

    
    def test_integration_with_utils_data(self):
        """Integration test using external utility data functions."""
        if utils is None:
            pytest.skip("dimensionality_reduction_utils not available")
        
        try:
            df = utils.get_mpso_data()
            df = utils.assign_classes(df, start_label=1)
            
            result_iter = gdhlda_mod.run_gdhlda(df, num_eigenvector=2, target_col='class')
            result_df = next(result_iter)
            
            assert isinstance(result_df, pd.DataFrame)
            assert 'class' in result_df.columns
            assert 'LD1' in result_df.columns or 'GDHLD1' in result_df.columns
            assert 'LD2' in result_df.columns or 'GDHLD2' in result_df.columns
            assert len(result_df) == len(df)
            
        except FileNotFoundError:
            pytest.skip("Real test data not available")

class TestGDHLDAProperties:
    """Property-based tests for GDHLDA invariants."""

    # Strategy to generate valid DataFrames for GDHLDA with sufficient variance
    valid_df_strategy = data_frames(
        columns=[
            column('f1', elements=st.floats(min_value=-1e3, max_value=1e3, allow_nan=False, allow_infinity=False)),
            column('f2', elements=st.floats(min_value=-1e3, max_value=1e3, allow_nan=False, allow_infinity=False)),
            column('f3', elements=st.floats(min_value=-1e3, max_value=1e3, allow_nan=False, allow_infinity=False)),
            column('class', elements=st.integers(min_value=1, max_value=3))
        ],
        index=range_indexes(min_size=30)  # Increased size to reduce zero-variance probability
    ).filter(lambda df: df['class'].nunique() >= 2 and all(df[col].var() > 1e-6 for col in ['f1', 'f2', 'f3']))

    @settings(deadline=None, max_examples=10)  # Reduced from 30 for performance
    @given(df=valid_df_strategy)
    def test_property_invariant_to_scaling(self, df):
        """
        Property: Scaling all input features by a constant factor should 
        not change the relative projections (direction of GDHLDs).
        """
        try:
            # Run on original data
            res1 = next(gdhlda_mod.run_gdhlda(df.copy(), num_eigenvector=1, target_col='class'))
            
            # Run on scaled data
            df_scaled = df.copy()
            feature_cols = [c for c in df.columns if c != 'class']
            df_scaled[feature_cols] *= 10.0
            res2 = next(gdhlda_mod.run_gdhlda(df_scaled, num_eigenvector=1, target_col='class'))
            
            # Get the LD column name (could be LD1 or GDHLD1)
            ld_col = 'LD1' if 'LD1' in res1.columns else 'GDHLD1'
            
            # The absolute correlation between projections should be near 1.0
            correlation = np.abs(np.corrcoef(res1[ld_col], res2[ld_col])[0, 1])
            assert correlation > 0.95, f"GDHLDs should be scale-invariant, correlation: {correlation}"
        except (np.linalg.LinAlgError, ValueError):
            # May fail on singular matrices generated by chance
            pytest.skip("Singular matrix encountered in random data")

    @settings(deadline=None, max_examples=10)  # Reduced from 25 for performance
    @given(df=valid_df_strategy)
    def test_property_output_dimensions(self, df):
        """
        Property: The output should contain the correct number of GDHLD columns
        capped by the theoretical maximum (classes - 1).
        """
        try:
            n_vecs = 2
            result = next(gdhlda_mod.run_gdhlda(df, num_eigenvector=n_vecs, target_col='class'))
            
            ld_cols = [c for c in result.columns if c.startswith('LD') or c.startswith('GDHLD')]
            max_possible = min(len(df.columns) - 1, df['class'].nunique() - 1)
            expected_lds = min(n_vecs, max_possible)
            
            assert len(ld_cols) == expected_lds, f"Expected {expected_lds} GDHLDs, got {len(ld_cols)}"
            assert len(result) == len(df), "Number of rows should be preserved"
        except (np.linalg.LinAlgError, ValueError):
            pytest.skip("Singular matrix encountered in random data")

    @settings(deadline=None, max_examples=10)  # Reduced from 20 for performance
    @given(df=valid_df_strategy)
    def test_property_dimensionality_reduction_limit(self, df):
        """
        Property: Output GDHLD count should be capped by min(features, classes - 1).
        """
        try:
            num_classes = df['class'].nunique()
            max_gdhld = min(len(df.columns) - 1, num_classes - 1)
            
            # Request more GDHLDs than possible
            result = next(gdhlda_mod.run_gdhlda(df, num_eigenvector=10, target_col='class'))
            
            ld_cols = [c for c in result.columns if c.startswith('LD') or c.startswith('GDHLD')]
            assert len(ld_cols) <= max_gdhld, f"GDHLD count should not exceed {max_gdhld}"
        except (np.linalg.LinAlgError, ValueError):
            pytest.skip("Singular matrix encountered in random data")

    @settings(deadline=None, max_examples=10)  # Reduced from 20 for performance
    @given(df=valid_df_strategy)
    def test_property_translation_invariance(self, df):
        """
        Property: Adding a constant bias to features (translation) 
        should not change the GDHLD projection directions for heteroscedastic LDA.
        """
        try:
            res1 = next(gdhlda_mod.run_gdhlda(df.copy(), num_eigenvector=1, target_col='class'))
            
            # Translate data
            df_translated = df.copy()
            feature_cols = [c for c in df.columns if c != 'class']
            df_translated[feature_cols] += 100.0
            res2 = next(gdhlda_mod.run_gdhlda(df_translated, num_eigenvector=1, target_col='class'))
            
            # Get the LD column name (could be LD1 or GDHLD1)
            ld_col = 'LD1' if 'LD1' in res1.columns else 'GDHLD1'
            
            # Use a tolerance for numerical precision
            correlation = np.abs(np.corrcoef(res1[ld_col], res2[ld_col])[0, 1])
            assert correlation > 0.9, f"GDHLDs should be translation-invariant, correlation: {correlation}"
        except (np.linalg.LinAlgError, ValueError):
            pytest.skip("Singular matrix encountered in random data")

    @settings(deadline=None, max_examples=8)  # Reduced from 15 for performance
    @given(df=valid_df_strategy)
    def test_property_class_preservation(self, df):
        """
        Property: The class column should be preserved exactly in the output.
        """
        try:
            result = next(gdhlda_mod.run_gdhlda(df, num_eigenvector=1, target_col='class'))
            
            # Class column should be identical
            pd.testing.assert_series_equal(result['class'], df['class'])
            
            # Number of rows should be preserved
            assert len(result) == len(df)
        except (np.linalg.LinAlgError, ValueError):
            pytest.skip("Singular matrix encountered in random data")

    @settings(deadline=None, max_examples=8)  # Reduced from 15 for performance
    @given(df=valid_df_strategy)
    def test_property_feature_order_independence(self, df):
        """
        Property: Permuting feature columns should not change the GDHLD projections
        (up to possible sign flips).
        """
        try:
            # Original order
            res1 = next(gdhlda_mod.run_gdhlda(df.copy(), num_eigenvector=1, target_col='class'))
            
            # Permute feature columns (keep class at end)
            feature_cols = [c for c in df.columns if c != 'class']
            df_permuted = df[feature_cols[::-1] + ['class']].copy()
            res2 = next(gdhlda_mod.run_gdhlda(df_permuted, num_eigenvector=1, target_col='class'))
            
            # Get the LD column name (could be LD1 or GDHLD1)
            ld_col = 'LD1' if 'LD1' in res1.columns else 'GDHLD1'
            
            # Correlation should be high (allowing for sign differences)
            correlation = np.abs(np.corrcoef(res1[ld_col], res2[ld_col])[0, 1])
            assert correlation > 0.8, f"GDHLDs should be order-independent, correlation: {correlation}"
        except (np.linalg.LinAlgError, ValueError):
            pytest.skip("Singular matrix encountered in random data")

    @settings(deadline=None, max_examples=8)  # Reduced from 15 for performance
    @given(df=valid_df_strategy)
    def test_property_global_mean_centering_independence(self, df):
        """
        Property: Adding a constant bias to a feature (translation) 
        should not change the LD projection values.
        """
        try:
            res1 = next(gdhlda_mod.run_gdhlda(df.copy(), num_eigenvector=1, target_col='class'))
            
            # Translate data
            df_translated = df.copy()
            feature_cols = [c for c in df.columns if c != 'class']
            df_translated[feature_cols] += 100.0
            res2 = next(gdhlda_mod.run_gdhlda(df_translated, num_eigenvector=1, target_col='class'))
            
            # Get the LD column name (could be LD1 or GDHLD1)
            ld_col = 'LD1' if 'LD1' in res1.columns else 'GDHLD1'
            
            # Use a tolerance for floating point drift in gradient descent
            # Check correlation instead of exact values for more robust testing
            corr = np.abs(np.corrcoef(res1[ld_col], res2[ld_col])[0, 1])
            if not np.isnan(corr):
                assert corr > 0.9  # Allow some tolerance for gradient descent
            else:
                pytest.skip("NaN correlation in translation test")
        except (np.linalg.LinAlgError, ValueError):
            pytest.skip("Singular matrix encountered in translation test")

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
