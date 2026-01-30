import os
import numpy as np
import pandas as pd
from pathlib import Path
import h5py

# BASE_DIR should be set to the base_directory_of_analysis mentioned in NEW_README.md
# It can be overriden by the DATA_BASE_DIR environment variable.
DATA_BASE_DIR = "/work/hdd/bfri/jjeong7/analysis_output/dist_maps"
BASE_DIR = os.environ.get("DATA_BASE_DIR", DATA_BASE_DIR)

# Default residues based on the 150-residue canonical mapping in 2.1.NEW_README.md
DEFAULT_RESIDUE_LIST = list(range(150))

def get_residue_feature_names(residue_list=DEFAULT_RESIDUE_LIST):
    """
    Generates pairwise feature names from a list of residue IDs (slots).
    Matches the condensed upper triangle order (i < j) used in the 150x150 distance matrix.
    Total features: 150*(149)/2 = 11,175.
    
    Args:
        residue_list (list): Sorted list of canonical residue slots (0-149).
        
    Returns:
        list: List of strings in 'RES<i>_<j>' format.
    """
    names = []
    n_res = len(residue_list)
    # The condensed upper triangle order in row-major corresponds to squareform
    for i in range(n_res):
        for j in range(i + 1, n_res):
            names.append(f"RES{residue_list[i] + 1}_{residue_list[j] + 1}")
    return names

def get_data_files(base_dir=BASE_DIR):
    """
    Finds all pairwise_dist.npy files and extracts metadata.
    
    Expected structure:
    base_dir/
        ├── construct/
        │   ├── subconstruct/
        │   │   ├── {replica}_s{start}_e{end}_pairwise_dist.npy
    """
    data_files = []
    base_path = Path(base_dir)
    
    # glob pattern to match the described file structure
    for npy_file in base_path.glob("**/*_pairwise_dist.npy"):
        # Relative path parts: (construct, subconstruct, filename)
        try:
            rel_path = npy_file.relative_to(base_path)
            parts = rel_path.parts
            
            if len(parts) >= 3:
                construct = parts[0]
                subconstruct = parts[1]
                filename = parts[-1]
                
                # Extract metadata from filename (e.g., "1_s0001_e0150_pairwise_dist.npy")
                file_parts = filename.split("_")
                replica_str = file_parts[0]
                
                # Parse start frame (e.g., "s0001" -> 1)
                start_frame = 1
                if len(file_parts) > 1 and file_parts[1].startswith("s"):
                    try:
                        start_frame = int(file_parts[1][1:])
                    except ValueError:
                        pass

                try:
                    replica = int(replica_str)
                except ValueError:
                    replica = replica_str
                
                data_files.append({
                    "path": str(npy_file),
                    "construct": construct,
                    "subconstruct": subconstruct,
                    "replica": replica,
                    "start_frame": start_frame
                })
        except ValueError:
            # Handle cases where path logic might fail if not under base_dir
            continue
            
    # Sort files by (construct, subconstruct, replica, start_frame)
    data_files.sort(key=lambda x: (x["construct"], x["subconstruct"], str(x["replica"]), x["start_frame"]))
    return data_files

def data_iterator(base_dir=BASE_DIR, chunk_size=None, residue_list=DEFAULT_RESIDUE_LIST, keep_features=None):
    """
    Iterative data provider that yields DataFrames with metadata.
    
    Args:
        base_dir: Root directory to search for data.
        chunk_size: If provided, yields DataFrames in chunks of this many rows.
        residue_list: List of residues used to generate pairwise distance columns.
        keep_features: List of feature names to keep. If None, keeps all.
    
    Yields:
        pd.DataFrame: A DataFrame with pairwise distance data, metadata, and frame numbers.
    """
    data_files = get_data_files(base_dir)
    feature_names = get_residue_feature_names(residue_list)
    
    if not data_files:
        print(f"No data files found in {base_dir}")
        return

    metadata_cols = ['construct', 'subconstruct', 'replica', 'frame_number']
    
    # Pre-compute columns to keep if specified
    final_cols = None
    if keep_features is not None:
        # Ensure metadata columns are always included
        final_cols = metadata_cols + [f for f in keep_features if f not in metadata_cols]

    for file_info in data_files:
        try:
            data = np.load(file_info["path"], mmap_mode='r')
            
            # Convert to DataFrame with interpretable feature names
            if data.shape[1] == len(feature_names):
                df = pd.DataFrame(data, columns=feature_names)
            else:
                # Fallback to integer columns if mismatch, but warn
                print(f"Warning: Data shape {data.shape[1]} doesn't match feature list {len(feature_names)} for {file_info['path']}")
                df = pd.DataFrame(data)
            
            # Assign metadata
            df['construct'] = file_info['construct']
            df['subconstruct'] = file_info['subconstruct']
            df['replica'] = file_info['replica']
            
            # Assign frame_number (1-indexed, starting from start_frame)
            df['frame_number'] = np.arange(len(df)) + file_info['start_frame']
            
            # Filter columns if requested
            if final_cols:
                # Only keep columns that exist in the dataframe
                # (Intersection to be safe against missing features)
                available_cols = [c for c in final_cols if c in df.columns]
                df = df[available_cols]

            if chunk_size:
                for i in range(0, len(df), chunk_size):
                    # We use .copy() to ensure we don't have a view of the original mmapped data 
                    # if we are doing further processing that might be affected.
                    yield df.iloc[i:i+chunk_size].copy()
            else:
                yield df
        except Exception as e:
            print(f"Error processing {file_info['path']}: {e}")
            continue

def load_h5_data(h5_path, dataset_name='data', chunk_size=None):
    """
    Load data from HDF5 file with optional chunking for large datasets.
    
    Args:
        h5_path (str): Path to .h5 file.
        dataset_name (str): Name of dataset within the .h5 file (default: 'data').
        chunk_size (int): If provided, yields chunks of this size. If None, loads all data.
    
    Yields (if chunk_size given):
        pd.DataFrame: Chunk of data.
    
    Returns (if chunk_size is None):
        pd.DataFrame: Full dataset (all data loaded into RAM).
    """
    if chunk_size is None:
        # Load everything at once
        return pd.read_hdf(h5_path, dataset_name)
    else:
        # Yield chunks to reduce peak RAM usage
        store = pd.HDFStore(h5_path, mode='r')
        n_rows = store.get_storer(dataset_name).nrows
        
        for start in range(0, n_rows, chunk_size):
            stop = min(start + chunk_size, n_rows)
            chunk = store.select(dataset_name, start=start, stop=stop)
            yield chunk
        
        store.close()


def save_h5_data(data, h5_path, dataset_name='data', mode='w', format='table'):
    """
    Save data to HDF5 file.
    
    Args:
        data (pd.DataFrame or np.ndarray): Data to save.
        h5_path (str): Path to output .h5 file.
        dataset_name (str): Name for dataset within .h5 file.
        mode (str): 'w' to overwrite, 'a' to append.
        format (str): 'fixed' (faster, no query) or 'table' (slower, queryable).
    """
    if isinstance(data, np.ndarray):
        data = pd.DataFrame(data)
    
    data.to_hdf(h5_path, dataset_name, mode=mode, format=format, complevel=9, complib='blosc')
    print(f"Saved {data.shape} to {h5_path}[{dataset_name}]")


def get_h5_info(h5_path):
    """
    Get basic info about .h5 file structure and dataset sizes.
    
    Args:
        h5_path (str): Path to .h5 file.
    
    Returns:
        dict: File information including datasets and their shapes.
    """
    info = {}
    with h5py.File(h5_path, 'r') as f:
        def visit_func(name, obj):
            if isinstance(obj, h5py.Dataset):
                info[name] = {
                    'shape': obj.shape,
                    'dtype': str(obj.dtype),
                    'size_mb': obj.nbytes / (1024 ** 2)
                }
        
        f.visititems(visit_func)
    
    return info


if __name__ == "__main__":
    # Quick test/validation script
    print(f"Searching in: {BASE_DIR}")
    files = get_data_files()
    print(f"Found {len(files)} data files.")
    
    if files:
        print("\nFirst 5 files found:")
        for f in files[:5]:
            print(f"  {f['construct']}/{f['subconstruct']} - Replica {f['replica']} -> {os.path.basename(f['path'])}")
        
        print("\nTesting iterator (first chunk):")
        for i, chunk in enumerate(data_iterator(chunk_size=10)):
            print(f"Chunk 0 shape: {chunk.shape}")
            print("Columns:", chunk.columns.tolist()[-5:])
            print("First row Sample:\n", chunk.iloc[0][-5:])
            break
