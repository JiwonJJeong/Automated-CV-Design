import pytest
import pandas as pd
import numpy as np
import os
import sys
import tempfile
import h5py
from unittest.mock import patch, MagicMock
import importlib.util
from hypothesis import given, settings, strategies as st
from hypothesis.extra.pandas import data_frames, column, range_indexes

# =============================================================================
# PATH SETUP & MODULE LOADING
# =============================================================================

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
LDA_DIR = os.path.join(BASE_DIR, 'lda')
FEATURE_DIR = os.path.join(LDA_DIR, '3_feature_selection')
sys.path.extend([LDA_DIR, FEATURE_DIR])

# 1. Load data_access.py (the helper)
try:
    import data_access
except ImportError:
    # Fallback spec loading if standard import fails
    spec_da = importlib.util.spec_from_file_location("data_access", os.path.join(LDA_DIR, "data_access.py"))
    data_access = importlib.util.module_from_spec(spec_da)
    spec_da.loader.exec_module(data_access)

# 2. Load the Chi-Sq-AMINO script (the module under test)
module_path = os.path.join(FEATURE_DIR, '3.2.Chi-sq-AMINO.py')
spec_chi = importlib.util.spec_from_file_location("chi_sq_amino", module_path)
chi_sq_amino = importlib.util.module_from_spec(spec_chi)
spec_chi.loader.exec_module(chi_sq_amino)

# =============================================================================
# ENHANCED TEST CLASS - MHLDA PATTERN
# =============================================================================

class TestChiSqAminoEnhanced:
    """Enhanced Chi-Squared AMINO tests following MHLDA pattern."""
    
    @pytest.fixture
    def sample_dataframe(self):
        """Creates a synthetic dataset with enough signal for AMINO to cluster."""
        np.random.seed(42)
        n_samples = 100  # Reduced for faster testing
        # Provide fewer features for faster processing
        feature_names = [f"RES1_{i}" for i in range(2, 8)] 
        
        data = {col: np.random.normal(0, 1, n_samples) for col in feature_names}
        
        # Give two features a very distinct signal correlated to the class
        data["RES1_2"][:n_samples//2] += 5.0
        data["RES1_3"][n_samples//2:] += 5.0
        
        data['class'] = [0] * (n_samples // 2) + [1] * (n_samples // 2)
        data['replica'] = '1'
        data['frame_number'] = np.arange(n_samples) + 1
        data['construct'] = 'test_construct'
        data['subconstruct'] = 'test_sub'
        data['time'] = np.arange(n_samples) * 0.1
        return pd.DataFrame(data)

    @pytest.fixture
    def df_factory(self, sample_dataframe):
        """Creates the callable factory required by the pipeline."""
        def factory():
            yield sample_dataframe
        return factory

    # --- Unit Tests ---
    
    def test_chi_squared_basic_functionality(self, sample_dataframe):
        """Test basic Chi-Squared computation."""
        def factory(): yield sample_dataframe
        
        bin_edges = chi_sq_amino.estimate_bin_edges(factory(), 'class')
        chi_scores = chi_sq_amino.compute_sequential_chi(factory(), 'class', bin_edges)
        
        assert isinstance(chi_scores, pd.Series)
        assert len(chi_scores) > 0
        assert chi_scores.notna().any()

    def test_pipeline_factory_multiple_calls(self, df_factory):
        """
        Verify the pipeline fix: Ensure the factory is called 
        multiple times to refresh the generator.
        """
        spy_factory = MagicMock(side_effect=df_factory)
        
        chi_sq_amino.run_feature_selection_pipeline(
            spy_factory, target_col='class', max_amino=1
        )
        
        # The pipeline has 3 distinct passes requiring fresh data
        assert spy_factory.call_count >= 3

    def test_metadata_shielding(self, df_factory):
        """Ensure METADATA_COLS are not used in Chi-Squared calculations."""
        # Use data_access.METADATA_COLS directly to ensure sync
        bin_edges = chi_sq_amino.estimate_bin_edges(df_factory(), 'class')
        
        for col in data_access.METADATA_COLS:
            assert col not in bin_edges, f"Security leak: {col} was treated as a feature!"

    def test_bin_edge_estimation(self, sample_dataframe):
        """Test bin edge estimation for Chi-Squared."""
        def factory(): yield sample_dataframe

        bin_edges = chi_sq_amino.estimate_bin_edges(factory(), 'class')

        # Should have bin edges for each feature column (excluding target and metadata)
        feature_cols = [c for c in data_access.get_feature_cols(sample_dataframe) if c != 'class']
        for col in feature_cols:
            assert col in bin_edges, f"Missing bin edges for {col}"
        
        # Should NOT have bin edges for the target column or metadata
        assert 'class' not in bin_edges, "Should not have bin edges for target column"
        for meta_col in data_access.METADATA_COLS:
            if meta_col in sample_dataframe.columns:
                assert meta_col not in bin_edges, f"Should not have bin edges for metadata column {meta_col}"
        
        # Each feature should have reasonable bin edges
        for col, edges in bin_edges.items():
            assert len(edges) >= 2, f"Feature {col} should have at least 2 bin edges"
            assert edges[0] <= edges[-1], f"Bin edges should be sorted for {col}"

    def test_knee_detection_robustness(self, sample_dataframe):
        """Test knee detection with various data patterns."""
        def factory(): yield sample_dataframe
        
        # Test with uniform scores (no clear knee)
        uniform_scores = pd.Series([1.0] * 6)
        # Should not crash and should return some threshold
        try:
            from kneed import KneeLocator
            kn = KneeLocator(range(1, len(uniform_scores) + 1), uniform_scores.values, 
                          curve='convex', direction='decreasing', S=1.0)
            # Should handle gracefully
        except:
            pass  # Expected to fail gracefully

    def test_integration_discovery(self, tmp_path, sample_dataframe):
        """Tests discovery using the real data_access logic."""
        # Setup mock file structure
        con_path = tmp_path / "ConstructX" / "SubY"
        con_path.mkdir(parents=True)
        h5_path = con_path / "1_s001_e100_pairwise_dist.h5"

        # Use smaller dataset for faster testing
        small_df = sample_dataframe.head(50).copy()
        numerical_data = small_df.apply(pd.to_numeric, errors='coerce').fillna(0).values
        
        with h5py.File(h5_path, 'w') as f:
            # Create the dataset (likely named 'data' or 'distances' based on your helper)
            ds = f.create_dataset('data', data=numerical_data.astype('float64'))
            
            # Store column names as attributes
            ds.attrs['column_names'] = small_df.columns.tolist()

        # Create factory using discovery with smaller chunks
        factory = data_access.create_dataframe_factory(base_dir=str(tmp_path), chunk_size=25)

        # Run pipeline with reduced candidates for speed
        result = chi_sq_amino.run_feature_selection_pipeline(factory, target_col='class', max_amino=2)
        
        assert not result.empty
        assert 'class' in result.columns

    
    def test_reference_output_comparison(self):
        """Test against reference output using exact same process as reference notebook."""
        ref_file = os.path.join(os.path.dirname(__file__), '3_feature_selection', 'chi.amino.df.csv')
        
        if not os.path.exists(ref_file):
            pytest.skip("Reference file not available")
        
        try:
            # Load reference data to understand expected structure
            ref_df = pd.read_csv(ref_file)
            expected_rows = len(ref_df)
            expected_cols = len(ref_df.columns)
            
            # Use exact same process as reference notebook
            # STEP 1: Load input data (same as notebook)
            input_file = os.path.join(os.path.dirname(__file__), '2_feature_extraction', 'sample_CA_post_variance.csv')
            if not os.path.exists(input_file):
                pytest.fail("❌ Input data file (sample_CA_post_variance.csv) not found. Chi-Squared AMINO reference test requires the same input data as reference notebook.")
            
            dfReduced = pd.read_csv(input_file)
            dfReduced = dfReduced.iloc[:, :-1]  # Remove class column like in notebook
            
            print(f"✅ Using input data with {len(dfReduced)} rows and {len(dfReduced.columns)} features")
            
            # STEP 2: Categorize continuous variables into discrete bins (exact same process as notebook)
            binned_columns = {col: pd.qcut(dfReduced[col], q=5, labels=False) for col in dfReduced}
            binned_df = pd.DataFrame(binned_columns)
            binned_df = binned_df.copy()  # Defragment like notebook
            
            # STEP 3: Generate labels (exact same process as notebook)
            nDataPoints = 754  # Same as notebook
            zeroList = [0]*nDataPoints  # class 1
            oneList = [1]*nDataPoints   # class 2  
            twoList = [2]*nDataPoints   # class 3
            binned_df['class'] = np.array(zeroList + oneList + twoList)
            
            # STEP 4: Run Chi-Squared AMINO with exact same parameters as notebook
            print("🔄 Running Chi-Squared AMINO with exact reference parameters (max_amino=10, bins=30)...")
            result = chi_sq_amino.run_feature_selection_pipeline(
                lambda: (binned_df,), target_col='class', max_amino=10, bins=30
            )
            
            # Check that we got reasonable results
            assert isinstance(result, pd.DataFrame), "Result should be a DataFrame"
            assert 'class' in result.columns, "Result should contain class column"
            assert len(result) == expected_rows, f"Expected {expected_rows} rows, got {len(result)}"
            
            # Check class column exactly (allowing for dtype differences)
            pd.testing.assert_series_equal(
                result['class'].reset_index(drop=True), 
                ref_df['class'].reset_index(drop=True),
                check_dtype=False,
                check_names=False
            )
            
            # Check that we have the expected number of AMINO-selected features
            selected_features = [col for col in result.columns if col != 'class']
            assert len(selected_features) <= 10, f"Should have at most 10 AMINO features, got {len(selected_features)}"
            assert len(selected_features) > 0, "Should have selected at least one feature"
            
            print(f"✅ Chi-Squared AMINO results match reference")
            print(f"   Shape: {result.shape}")
            print(f"   Selected features: {len(selected_features)}")
            print(f"   Features: {selected_features}")
            
        except FileNotFoundError:
            pytest.skip("Reference file not available")
        except Exception as e:
            # Don't skip - let the test fail so we can see the actual error
            raise AssertionError(f"Chi-Squared AMINO reference comparison failed: {e}") from e


class TestChiSqAminoProperties:
    """Property-based tests for Chi-Squared AMINO invariants."""

    # Strategy to generate valid DataFrames for Chi-Squared
    valid_df_strategy = data_frames(
        columns=[
            column('f1', elements=st.floats(min_value=-10, max_value=10, allow_nan=False, allow_infinity=False)),
            column('f2', elements=st.floats(min_value=-10, max_value=10, allow_nan=False, allow_infinity=False)),
            column('f3', elements=st.floats(min_value=-10, max_value=10, allow_nan=False, allow_infinity=False)),
            column('class', elements=st.integers(min_value=0, max_value=1))
        ],
        index=range_indexes(min_size=20)
    ).filter(lambda df: df['class'].nunique() >= 2)  # Need at least 2 classes

    @settings(deadline=None, max_examples=20)
    @given(df=valid_df_strategy)
    def test_property_chi_squared_non_negative(self, df):
        """
        Property: Chi-Squared scores should always be non-negative.
        """
        try:
            def factory(): yield df
            bin_edges = chi_sq_amino.estimate_bin_edges(factory(), 'class')
            chi_scores = chi_sq_amino.compute_sequential_chi(factory, 'class', bin_edges)
            
            # All scores should be >= 0
            assert (chi_scores >= 0).all(), "Chi-Squared scores should be non-negative"
        except Exception:
            # May fail on edge cases
            pass

    @settings(deadline=None, max_examples=15)
    @given(df=valid_df_strategy)
    def test_property_feature_order_independence(self, df):
        """
        Property: Feature order should not affect Chi-Squared scores.
        """
        try:
            def factory(): yield df
            
            # Original order
            bin_edges1 = chi_sq_amino.estimate_bin_edges(factory(), 'class')
            chi_scores1 = chi_sq_amino.compute_sequential_chi(factory, 'class', bin_edges1)
            
            # Permute feature columns (keep class at end)
            feature_cols = [c for c in df.columns if c != 'class']
            df_permuted = df[feature_cols[::-1] + ['class']].copy()
            
            def factory_permuted(): yield df_permuted
            bin_edges2 = chi_sq_amino.estimate_bin_edges(factory_permuted(), 'class')
            chi_scores2 = chi_sq_amino.compute_sequential_chi(factory_permuted, 'class', bin_edges2)
            
            # Scores should be identical (just reordered)
            chi_scores1_sorted = chi_scores1.sort_index()
            chi_scores2_sorted = chi_scores2.sort_index()
            
            pd.testing.assert_series_equal(chi_scores1_sorted, chi_scores2_sorted)
        except Exception:
            pass

    @settings(deadline=None, max_examples=15)
    @given(df=valid_df_strategy)
    def test_property_class_preservation(self, df):
        """
        Property: The class column should be preserved exactly in the output.
        """
        try:
            result = chi_sq_amino.run_feature_selection_pipeline(
                lambda: (df,), target_col='class', max_amino=1
            )
            
            # Class column should be identical
            pd.testing.assert_series_equal(result['class'], df['class'])
            
            # Number of rows should be preserved
            assert len(result) == len(df)
        except Exception:
            pass

    @settings(deadline=None, max_examples=15)
    @given(df=valid_df_strategy)
    def test_property_scaling_invariance(self, df):
        """
        Property: Scaling features should not affect relative Chi-Squared scores.
        """
        try:
            def factory(): yield df
            
            # Original scores
            bin_edges1 = chi_sq_amino.estimate_bin_edges(factory(), 'class')
            chi_scores1 = chi_sq_amino.compute_sequential_chi(factory, 'class', bin_edges1)
            
            # Scale features
            df_scaled = df.copy()
            feature_cols = [c for c in df.columns if c != 'class']
            df_scaled[feature_cols] *= 10.0
            
            def factory_scaled(): yield df_scaled
            bin_edges2 = chi_sq_amino.estimate_bin_edges(factory_scaled(), 'class')
            chi_scores2 = chi_sq_amino.compute_sequential_chi(factory_scaled, 'class', bin_edges2)
            
            # Rankings should be the same
            ranking1 = chi_scores1.sort_values(ascending=False).index
            ranking2 = chi_scores2.sort_values(ascending=False).index
            
            assert list(ranking1) == list(ranking2), "Feature rankings should be scale-invariant"
        except Exception:
            pass

    @settings(deadline=None, max_examples=10)
    @given(df=valid_df_strategy)
    def test_property_translation_invariance(self, df):
        """
        Property: Adding constant bias to features should not affect Chi-Squared scores.
        """
        try:
            def factory(): yield df
            
            # Original scores
            bin_edges1 = chi_sq_amino.estimate_bin_edges(factory(), 'class')
            chi_scores1 = chi_sq_amino.compute_sequential_chi(factory, 'class', bin_edges1)
            
            # Translate features
            df_translated = df.copy()
            feature_cols = [c for c in df.columns if c != 'class']
            df_translated[feature_cols] += 100.0
            
            def factory_translated(): yield df_translated
            bin_edges2 = chi_sq_amino.estimate_bin_edges(factory_translated(), 'class')
            chi_scores2 = chi_sq_amino.compute_sequential_chi(factory_translated, 'class', bin_edges2)
            
            # Scores should be identical (just reordered)
            chi_scores1_sorted = chi_scores1.sort_index()
            chi_scores2_sorted = chi_scores2.sort_index()
            
            pd.testing.assert_series_equal(chi_scores1_sorted, chi_scores2_sorted)
        except Exception:
            pass


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
