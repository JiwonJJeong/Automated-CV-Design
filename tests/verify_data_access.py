import sys
import os
import numpy as np
import pandas as pd

# Add the directory containing data_access.py to the path
sys.path.append(os.path.abspath('lda'))
import data_access as da

def test_discovery():
    print("\n--- Testing File Discovery ---")
    local_data = 'data/dist_maps'
    if os.path.exists(local_data):
        print(f"Using local data directory: {local_data}")
        files = da.get_data_files(base_dir=local_data)
    else:
        files = da.get_data_files()
    
    print(f"Total data files found: {len(files)}")
    
    h5_count = sum(1 for f in files if f['type'] == 'h5')
    npy_count = sum(1 for f in files if f['type'] == 'npy')
    
    print(f"  HDF5 files: {h5_count}")
    print(f"  NPY files:  {npy_count}")
    
    if len(files) > 0:
        print(f"  Sample file: {files[0]['path']} (Type: {files[0]['type']}, Replica: {files[0]['replica']})")
    
    return files

def test_iterator():
    print("\n--- Testing Data Iterator ---")
    local_data = 'data/dist_maps'
    # Small chunk size for testing
    if os.path.exists(local_data):
        it = da.data_iterator(base_dir=local_data, chunk_size=100)
    else:
        it = da.data_iterator(chunk_size=100)
    
    try:
        df = next(it)
        print(f"First chunk shape: {df.shape}")
        
        expected_meta = ['construct', 'subconstruct', 'replica', 'frame_number']
        missing = [c for c in expected_meta if c not in df.columns]
        
        if not missing:
            print("  [SUCCESS] All metadata columns present.")
        else:
            print(f"  [FAILURE] Missing columns: {missing}")
            
        if 'time' in df.columns:
            print(f"  [SUCCESS] 'time' column is present.")
        else:
            print("  [INFO] 'time' column not found (expected if time.npy is missing).")
            
        print(f"  Replica info: {df['replica'].unique()}")
        print(f"  Construct info: {df['construct'].unique()}")
        
    except StopIteration:
        print("  [WARNING] Iterator yielded no data.")
    except Exception as e:
        print(f"  [ERROR] Iterator failed: {e}")

def test_conversion(npy_files):
    if not npy_files:
        print("\n--- Skipping Conversion Test (No NPY files found) ---")
        return

    print("\n--- Testing NPY to H5 Conversion ---")
    src = npy_files[0]['path']
    dst = src.replace('.npy', '_TEST_CONV.h5')
    
    try:
        da.convert_npy_to_h5(src, dst)
        if os.path.exists(dst):
            print(f"  [SUCCESS] Created {dst}")
            
            # Verify the converted file
            df = da.load_h5_data(dst)
            print(f"  Loaded converted data shape: {df.shape}")
            
            # Clean up
            os.remove(dst)
            print("  [CLEANUP] Deleted test H5 file.")
        else:
            print("  [FAILURE] Conversion did not produce a file.")
    except Exception as e:
        print(f"  [ERROR] Conversion failed: {e}")

if __name__ == "__main__":
    files = test_discovery()
    test_iterator()
    npy_files = [f for f in files if f['type'] == 'npy']
    test_conversion(npy_files)
