import pytest
import pandas as pd
import numpy as np
import os
import sys
import tempfile
import h5py
from unittest.mock import patch, MagicMock
import importlib.util

# =============================================================================
# PATH SETUP & MODULE LOADING
# =============================================================================

# Define the project root based on the test file location
# tests/ -> project_root/
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# Add the lda directory to the path so we can find data_access.py
LDA_DIR = os.path.join(BASE_DIR, 'lda')
sys.path.append(LDA_DIR)

# Also add the specific feature selection directory for the script under test
FEATURE_DIR = os.path.join(LDA_DIR, '3_feature_selection')
sys.path.append(FEATURE_DIR)

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
# TEST CLASS
# =============================================================================

class TestChiSqAmino:
    
    @pytest.fixture
    def sample_dataframe(self):
        """Creates a synthetic dataset with enough signal for AMINO to cluster."""
        np.random.seed(42)
        n_samples = 400  # Increased for better PDF estimation
        # Provide more features so AMINO doesn't collapse clusters
        feature_names = [f"RES1_{i}" for i in range(2, 12)] 
        
        data = {col: np.random.normal(0, 1, n_samples) for col in feature_names}
        
        # Give two features a very distinct signal correlated to the class
        data["RES1_2"][:n_samples//2] += 5.0
        data["RES1_3"][n_samples//2:] += 5.0
        
        data['class'] = [0] * (n_samples // 2) + [1] * (n_samples // 2)
        data['replica'] = '1'
        data['frame_number'] = np.arange(n_samples) + 1
        return pd.DataFrame(data)

    @pytest.fixture
    def df_factory(self, sample_dataframe):
        """Creates the callable factory required by the pipeline."""
        def factory():
            yield sample_dataframe
        return factory

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

    def test_integration_discovery(self, tmp_path, sample_dataframe):
        """Tests discovery using the real data_access logic."""
        # Setup mock file structure
        con_path = tmp_path / "ConstructX" / "SubY"
        con_path.mkdir(parents=True)
        h5_path = con_path / "1_s001_e100_pairwise_dist.h5"

        # 1. Separate the data: h5py needs numerical arrays
        # We convert everything to float64. Ensure 'class' is numeric (0, 1) not strings.
        numerical_data = sample_dataframe.apply(pd.to_numeric, errors='coerce').fillna(0).values
        
        with h5py.File(h5_path, 'w') as f:
            # Create the dataset (likely named 'data' or 'distances' based on your helper)
            ds = f.create_dataset('data', data=numerical_data.astype('float64'))
            
            # 2. Store column names as attributes
            # HDF5 attributes handle strings better than the dataset itself
            ds.attrs['column_names'] = sample_dataframe.columns.tolist()

        # 3. Create factory using discovery
        factory = data_access.create_dataframe_factory(base_dir=str(tmp_path), chunk_size=50)

        # 4. Run pipeline
        result = chi_sq_amino.run_feature_selection_pipeline(factory, target_col='class')

        assert not result.empty
        assert 'class' in result.columns

if __name__ == "__main__":
    pytest.main([__file__, "-v"])