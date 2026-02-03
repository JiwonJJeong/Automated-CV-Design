import pytest
import pandas as pd
import numpy as np
import os
import sys
import h5py
from unittest.mock import MagicMock
import importlib.util

# =============================================================================
# PATH SETUP & MODULE LOADING
# =============================================================================

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
LDA_DIR = os.path.join(BASE_DIR, 'lda')
FEATURE_DIR = os.path.join(LDA_DIR, '3_feature_selection')

# Add to sys.path for standard imports
sys.path.extend([LDA_DIR, FEATURE_DIR])

# 1. Load data_access.py
try:
    import data_access
except ImportError:
    spec_da = importlib.util.spec_from_file_location("data_access", os.path.join(LDA_DIR, "data_access.py"))
    data_access = importlib.util.module_from_spec(spec_da)
    spec_da.loader.exec_module(data_access)

# 2. Load Fisher-AMINO script 
module_path = os.path.join(FEATURE_DIR, '3.3.Fisher-AMINO.py') 
spec_f = importlib.util.spec_from_file_location("fisher_amino", module_path)
fisher_amino = importlib.util.module_from_spec(spec_f)
spec_f.loader.exec_module(fisher_amino)

# =============================================================================
# TEST CLASS
# =============================================================================

class TestFisherAmino:
    
    @pytest.fixture
    def sample_dataframe(self):
        """Creates a synthetic dataset with high-signal features for Fisher scoring."""
        np.random.seed(42)
        n_samples = 400
        # Create 10 features
        feature_names = [f"RES1_{i}" for i in range(2, 12)] 
        data = {col: np.random.normal(0, 1, n_samples) for col in feature_names}
        
        # Inject strong Fisher signal: Feature 2 is highly discriminative
        data["RES1_2"][:n_samples//2] += 10.0  # Class 0
        data["RES1_2"][n_samples//2:] -= 10.0  # Class 1
        
        data['class'] = [0] * (n_samples // 2) + [1] * (n_samples // 2)
        data['replica'] = '1'
        data['frame_number'] = np.arange(n_samples) + 1
        return pd.DataFrame(data)

    @pytest.fixture
    def df_factory(self, sample_dataframe):
        """Creates the callable factory required for sequential processing."""
        def factory():
            yield sample_dataframe
        return factory

    def test_sequential_fisher_logic(self, df_factory):
        """Verify Fisher scores identify the correct high-signal feature."""
        scores = fisher_amino.compute_sequential_fisher(df_factory, 'class')
        
        assert isinstance(scores, pd.Series)
        # RES1_2 should be the top feature due to injected signal
        assert scores.index[0] == "RES1_2"
        assert scores["RES1_2"] > scores["RES1_3"]

    def test_pipeline_factory_exhaustion_protection(self, df_factory):
        """Ensure the pipeline calls the factory multiple times (Pass 1 and Pass 2)."""
        spy_factory = MagicMock(side_effect=df_factory)
        
        fisher_amino.run_fisher_amino_pipeline(
            spy_factory, target_col='class', max_outputs=1
        )
        
        # Needs at least 2 calls: one for Fisher Scores, one for Extraction
        assert spy_factory.call_count >= 2

    def test_metadata_shielding(self, sample_dataframe):
        """Ensure metadata columns are not processed as features."""
        # Setup factory that includes metadata
        def factory(): yield sample_dataframe
        
        scores = fisher_amino.compute_sequential_fisher(factory, 'class')
        
        for meta in data_access.METADATA_COLS:
            assert meta not in scores.index, f"Metadata leak: {meta} was scored!"

    def test_integration_discovery_h5(self, tmp_path, sample_dataframe):
        """Full integration test using HDF5 discovery."""
        # 1. Setup mock H5 directory structure
        con_path = tmp_path / "ConstructX" / "SubY"
        con_path.mkdir(parents=True)
        
        # CHANGE: Use the naming convention data_access.py looks for
        h5_path = con_path / "1_s001_e100_pairwise_dist.h5" 

        # 2. Write H5 data
        numerical_data = sample_dataframe.apply(pd.to_numeric, errors='coerce').fillna(0).values
        with h5py.File(h5_path, 'w') as f:
            # Note: Ensure the dataset key 'data' matches what data_access expects
            ds = f.create_dataset('data', data=numerical_data.astype('float64'))
            ds.attrs['column_names'] = sample_dataframe.columns.tolist()

        # 3. Create factory
        factory = data_access.create_dataframe_factory(base_dir=str(tmp_path), chunk_size=100)

        # 4. Run pipeline
        result = fisher_amino.run_fisher_amino_pipeline(factory, target_col='class', max_outputs=2)

        assert not result.empty
        assert 'class' in result.columns

if __name__ == "__main__":
    pytest.main([__file__, "-v"])