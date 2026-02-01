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
    Finds all pairwise_dist.h5 files and extracts metadata.
    
    Expected structure:
    base_dir/
        ├── construct/
        │   ├── subconstruct/
        │   │   ├── {replica}_pairwise_dist.h5
    """
    data_files = []
    base_path = Path(base_dir)
    
    # glob pattern to match .h5 files
    for h5_file in base_path.glob("**/*_pairwise_dist.h5"):
        # Relative path parts: (construct, subconstruct, filename)
        try:
            rel_path = h5_file.relative_to(base_path)
            parts = rel_path.parts
            
            if len(parts) >= 3:
                construct = parts[0]
                subconstruct = parts[1]
                filename = parts[-1]
                
                # Extract replica from filename (e.g., "1_pairwise_dist.h5")
                file_parts = filename.split("_")
                replica_str = file_parts[0]

                try:
                    replica = int(replica_str)
                except ValueError:
                    replica = replica_str
                
                data_files.append({
                    "path": str(h5_file),
                    "construct": construct,
                    "subconstruct": subconstruct,
                    "replica": replica
                })
        except ValueError:
            # Handle cases where path logic might fail if not under base_dir
            continue
            
    # Sort files by (construct, subconstruct, replica)
    data_files.sort(key=lambda x: (x["construct"], x["subconstruct"], str(x["replica"])))
    return data_files

def data_iterator(base_dir=BASE_DIR, chunk_size=10000, dataset_name='data'):
    """
    Iterative data provider that yields DataFrames from H5 files with metadata.
    Reads H5 files chunk-by-chunk to avoid loading entire dataset into RAM.
    
    Args:
        base_dir: Root directory to search for .h5 files.
        chunk_size: Number of rows to read per chunk (default: 10000).
        dataset_name: Name of the dataset within each H5 file (default: 'data').
    
    Yields:
        pd.DataFrame: A DataFrame chunk with pairwise distance data and metadata columns
                     (construct, subconstruct, replica, frame_number).
    """
    data_files = get_data_files(base_dir)
    
    if not data_files:
        print(f"No .h5 data files found in {base_dir}")
        return

    for file_info in data_files:
        try:
            with h5py.File(file_info["path"], 'r') as f:
                if dataset_name not in f:
                    print(f"Warning: Dataset '{dataset_name}' not found in {file_info['path']}")
                    continue
                
                dataset = f[dataset_name]
                total_rows = dataset.shape[0]
                
                # Get column names from attributes if available
                column_names = dataset.attrs.get('column_names', None)
                if column_names is None:
                    # Generate default column names
                    column_names = [f'feature_{i}' for i in range(dataset.shape[1])]
                
                # Read and yield data in chunks
                for start_idx in range(0, total_rows, chunk_size):
                    end_idx = min(start_idx + chunk_size, total_rows)
                    
                    # Read chunk from H5 file
                    chunk_data = dataset[start_idx:end_idx]
                    
                    # Create DataFrame
                    df = pd.DataFrame(chunk_data, columns=column_names)
                    
                    # Add metadata columns
                    df['construct'] = file_info['construct']
                    df['subconstruct'] = file_info['subconstruct']
                    df['replica'] = file_info['replica']
                    df['frame_number'] = np.arange(start_idx, end_idx) + 1  # 1-indexed
                    
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
