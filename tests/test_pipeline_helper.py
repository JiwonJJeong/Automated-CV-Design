"""
Tests for pipeline helper functions, specifically class assignment logic.
"""

import pytest
import pandas as pd
import numpy as np
import os
import sys

# Add the lda directory to the path
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
LDA_DIR = os.path.join(BASE_DIR, 'lda')
sys.path.insert(0, LDA_DIR)

import pipeline_helper


class TestPipelineHelper:
    """Tests for pipeline helper functions."""

    def test_add_target_labels_basic_functionality(self):
        """Test basic functionality of add_target_labels."""
        # Create test data with known construct/subconstruct combinations
        test_data = pd.DataFrame({
            'RES1_2': np.random.normal(0, 1, 20),
            'RES1_3': np.random.normal(0, 1, 20),
            'construct': ['calmodulin'] * 10 + ['calmodulin-compact'] * 10,
            'subconstruct': ['ca-mg-1-2'] * 5 + ['ca-mg-1-4'] * 5 + ['ca-only'] * 5 + ['mg-only'] * 5,
            'replica': [1] * 20,
            'frame_number': range(1, 21),
            'time': np.arange(20) * 0.1
        })
        
        # Create iterator
        def df_iter():
            yield test_data
        
        # Apply class assignment
        result_iter = pipeline_helper.add_target_labels(df_iter())
        result_df = next(result_iter)
        
        # Verify class column was added
        assert 'class' in result_df.columns
        
        # Verify correct class assignments
        expected_classes = {
            ('calmodulin', 'ca-mg-1-2'): 1,
            ('calmodulin', 'ca-mg-1-4'): 2,
            ('calmodulin', 'ca-only'): 3,
            ('calmodulin', 'mg-only'): 4,
            ('calmodulin-compact', 'ca-mg-1-2'): 5,
            ('calmodulin-compact', 'ca-mg-1-4'): 6,
            ('calmodulin-compact', 'ca-only'): 7,
            ('calmodulin-compact', 'mg-only'): 8,
        }
        
        for _, row in result_df.iterrows():
            expected_class = expected_classes.get((row['construct'], row['subconstruct']), 0)
            assert row['class'] == expected_class, f"Class mismatch for {row['construct']} x {row['subconstruct']}"

    def test_add_target_labels_unknown_combinations(self):
        """Test handling of unknown construct/subconstruct combinations."""
        # Create test data with unknown combinations
        test_data = pd.DataFrame({
            'RES1_2': np.random.normal(0, 1, 10),
            'construct': ['unknown_construct'] * 5 + ['calmodulin'] * 5,
            'subconstruct': ['unknown_subconstruct'] * 5 + ['unknown_subconstruct'] * 5,
            'replica': [1] * 10,
            'frame_number': range(1, 11),
            'time': np.arange(10) * 0.1
        })
        
        def df_iter():
            yield test_data
        
        result_iter = pipeline_helper.add_target_labels(df_iter())
        result_df = next(result_iter)
        
        # Unknown combinations should get class 0
        assert all(result_df[result_df['construct'] == 'unknown_construct']['class'] == 0)
        assert all(result_df[result_df['subconstruct'] == 'unknown_subconstruct']['class'] == 0)

    def test_add_target_labels_multiple_chunks(self):
        """Test class assignment across multiple data chunks."""
        # Create multiple chunks
        chunk1 = pd.DataFrame({
            'RES1_2': np.random.normal(0, 1, 10),
            'construct': ['calmodulin'] * 10,
            'subconstruct': ['ca-mg-1-2'] * 10,
            'replica': [1] * 10,
            'frame_number': range(1, 11),
            'time': np.arange(10) * 0.1
        })
        
        chunk2 = pd.DataFrame({
            'RES1_2': np.random.normal(0, 1, 10),
            'construct': ['calmodulin-compact'] * 10,
            'subconstruct': ['ca-only'] * 10,
            'replica': [2] * 10,
            'frame_number': range(11, 21),
            'time': np.arange(10, 20) * 0.1
        })
        
        def df_iter():
            yield chunk1
            yield chunk2
        
        result_iter = pipeline_helper.add_target_labels(df_iter())
        result_chunks = list(result_iter)
        
        # Verify both chunks have correct class assignments
        assert len(result_chunks) == 2
        assert all(result_chunks[0]['class'] == 1)  # calmodulin + ca-mg-1-2 = class 1
        assert all(result_chunks[1]['class'] == 7)  # calmodulin-compact + ca-only = class 7

    def test_add_target_labels_class_distribution(self):
        """Test that all 8 expected classes are generated."""
        # Create test data with all combinations
        constructs = ['calmodulin', 'calmodulin-compact']
        subconstructs = ['ca-mg-1-2', 'ca-mg-1-4', 'ca-only', 'mg-only']
        
        data = []
        for construct in constructs:
            for subconstruct in subconstructs:
                for i in range(2):  # 2 samples per combination
                    data.append({
                        'RES1_2': np.random.normal(0, 1),
                        'construct': construct,
                        'subconstruct': subconstruct,
                        'replica': 1,
                        'frame_number': len(data) + 1,
                        'time': len(data) * 0.1
                    })
        
        test_df = pd.DataFrame(data)
        
        def df_iter():
            yield test_df
        
        result_iter = pipeline_helper.add_target_labels(df_iter())
        result_df = next(result_iter)
        
        # Verify all 8 classes are present
        unique_classes = sorted(result_df['class'].unique())
        expected_classes = list(range(1, 9))  # Classes 1-8
        assert unique_classes == expected_classes, f"Expected classes {expected_classes}, got {unique_classes}"
        
        # Verify each class has exactly 2 samples
        for class_id in expected_classes:
            class_count = (result_df['class'] == class_id).sum()
            assert class_count == 2, f"Class {class_id} should have 2 samples, got {class_count}"

    def test_add_target_labels_preserves_other_columns(self):
        """Test that non-class columns are preserved unchanged."""
        original_data = pd.DataFrame({
            'RES1_2': [1.0, 2.0, 3.0],
            'RES1_3': [4.0, 5.0, 6.0],
            'construct': ['calmodulin', 'calmodulin-compact', 'calmodulin'],
            'subconstruct': ['ca-mg-1-2', 'ca-only', 'mg-only'],
            'replica': [1, 2, 3],
            'frame_number': [10, 20, 30],
            'time': [1.0, 2.0, 3.0]
        })
        
        def df_iter():
            yield original_data
        
        result_iter = pipeline_helper.add_target_labels(df_iter())
        result_df = next(result_iter)
        
        # Verify all original columns are preserved
        for col in original_data.columns:
            assert col in result_df.columns, f"Column {col} was not preserved"
        
        # Verify values are unchanged (except for added class column)
        for col in original_data.columns:
            pd.testing.assert_series_equal(
                original_data[col].reset_index(drop=True),
                result_df[col].reset_index(drop=True),
                check_names=False
            )
        
        # Verify class column was added
        assert 'class' in result_df.columns
        assert len(result_df.columns) == len(original_data.columns) + 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
