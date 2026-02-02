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
    Finds all pairwise_dist.h5 and pairwise_dist.npy files and extracts metadata.
    Prefer .h5 if both exist for the same replica.
    
    Expected structure:
    base_dir/
    └── construct/
        └── subconstruct/
            └── {replica}_s{start}_e{end}_pairwise_dist.{h5|npy}
    """
    data_files_dict = {} # key: (construct, subconstruct, replica_str)
    base_path = Path(base_dir)
    
    # glob pattern to match .h5 and .npy files
    for data_file in base_path.glob("**/*_pairwise_dist.*"):
        if data_file.suffix not in ['.h5', '.npy']:
            continue
            
        try:
            rel_path = data_file.relative_to(base_path)
            parts = rel_path.parts
            
            if len(parts) >= 3:
                construct = parts[0]
                subconstruct = parts[1]
                filename = parts[-1]
                
                # Extract replica from filename (e.g., "1_s0001_e0300_pairwise_dist.h5")
                replica_str = filename.split("_")[0]
                
                key = (construct, subconstruct, replica_str)
                
                # If we already have an entry and the current one is .npy, skip it (prefer .h5)
                if key in data_files_dict and data_file.suffix == '.npy':
                    continue
                
                # If current is .h5, it will overwrite any existing .npy entry for same key
                data_files_dict[key] = {
                    "path": str(data_file),
                    "construct": construct,
                    "subconstruct": subconstruct,
                    "replica": replica_str,
                    "type": data_file.suffix[1:] # 'h5' or 'npy'
                }
        except ValueError:
            continue
            
    data_files = list(data_files_dict.values())
    # Sort files by (construct, subconstruct, replica_str)
    data_files.sort(key=lambda x: (x["construct"], x["subconstruct"], x["replica"]))
    return data_files

def convert_npy_to_h5(npy_path, h5_path, time_path=None, dataset_name='distances'):
    """
    Converts a legacy .npy distance file to the new HDF5 format.
    """
    print(f"Converting {npy_path} -> {h5_path}...")
    distances = np.load(npy_path)
    
    # Try to load companion time data
    times = None
    if time_path and os.path.exists(time_path):
        times = np.load(time_path)
    else:
        # Check in the same directory for time.npy
        potential_time_npy = os.path.join(os.path.dirname(npy_path), 'time.npy')
        if os.path.exists(potential_time_npy):
            times = np.load(potential_time_npy)
    
    with h5py.File(h5_path, 'w') as f:
        f.create_dataset(dataset_name, data=distances, compression="gzip", chunks=True)
        if times is not None:
            # If time.npy is a master file, slice it to match distances
            if len(times) >= len(distances):
                # Try to extract the range from filename if it's there
                # e.g., 0_s0001_e0300_...
                try:
                    parts = os.path.basename(npy_path).split('_')
                    if len(parts) > 2 and parts[1].startswith('s') and parts[2].startswith('e'):
                        start_idx = int(parts[1][1:]) - 1
                        end_idx = int(parts[2][1:])
                        f.create_dataset('times', data=times[start_idx:end_idx], compression="gzip")
                    else:
                        f.create_dataset('times', data=times[:len(distances)], compression="gzip")
                except:
                    f.create_dataset('times', data=times[:len(distances)], compression="gzip")
            else:
                f.create_dataset('times', data=times, compression="gzip")
    print("Conversion complete.")

def data_iterator(base_dir=BASE_DIR, chunk_size=10000, dataset_name='distances'):
    """
    Iterative data provider that yields DataFrames from H5 or NPY files.
    """
    data_files = get_data_files(base_dir)
    
    if not data_files:
        print(f"No data files found in {base_dir}")
        return

    for file_info in data_files:
        try:
            if file_info["type"] == 'h5':
                with h5py.File(file_info["path"], 'r') as f:
                    if dataset_name not in f:
                        # Fallback to 'data' if 'distances' isn't there
                        actual_ds = 'data' if 'data' in f else None
                        if not actual_ds:
                            print(f"Warning: No valid dataset found in {file_info['path']}")
                            continue
                    else:
                        actual_ds = dataset_name
                    
                    dataset = f[actual_ds]
                    total_rows = dataset.shape[0]
                    times = f['times'][:] if 'times' in f else None
                    column_names = dataset.attrs.get('column_names', [f'feature_{i}' for i in range(dataset.shape[1])])
                    
                    for start_idx in range(0, total_rows, chunk_size):
                        end_idx = min(start_idx + chunk_size, total_rows)
                        chunk_data = dataset[start_idx:end_idx]
                        df = pd.DataFrame(chunk_data, columns=column_names)
                        
                        df['construct'] = file_info['construct']
                        df['subconstruct'] = file_info['subconstruct']
                        df['replica'] = file_info['replica']
                        if times is not None:
                            df['time'] = times[start_idx:end_idx]
                        df['frame_number'] = np.arange(start_idx, end_idx) + 1
                        yield df
            
            elif file_info["type"] == 'npy':
                # Map-read the npy for memory efficiency
                data = np.load(file_info["path"], mmap_mode='r')
                total_rows = data.shape[0]
                column_names = [f'feature_{i}' for i in range(data.shape[1])]
                
                # Mock times if time.npy exists
                times = None
                potential_time_npy = os.path.join(os.path.dirname(file_info["path"]), 'time.npy')
                if os.path.exists(potential_time_npy):
                    full_times = np.load(potential_time_npy)
                    # Try to slice based on filename
                    try:
                        pts = os.path.basename(file_info["path"]).split('_')
                        s = int(pts[1][1:]) - 1
                        e = int(pts[2][1:])
                        times = full_times[s:e]
                    except:
                        times = full_times[:total_rows]

                for start_idx in range(0, total_rows, chunk_size):
                    end_idx = min(start_idx + chunk_size, total_rows)
                    chunk_data = data[start_idx:end_idx]
                    df = pd.DataFrame(chunk_data, columns=column_names)
                    
                    df['construct'] = file_info['construct']
                    df['subconstruct'] = file_info['subconstruct']
                    df['replica'] = file_info['replica']
                    if times is not None:
                        df['time'] = times[start_idx:end_idx]
                    df['frame_number'] = np.arange(start_idx, end_idx) + 1
                    yield df
                    
        except Exception as e:
            print(f"Error processing {file_info['path']}: {e}")
            continue

def load_h5_data(h5_path, dataset_name='distances', chunk_size=None):
    """
    Load data from HDF5 file with optional chunking.
    Supports both pandas.to_hdf and raw h5py datasets.
    """
    if chunk_size is not None:
        return load_h5_data_iterator(h5_path, dataset_name, chunk_size)
    
    try:
        # Try pandas first
        return pd.read_hdf(h5_path, dataset_name)
    except:
        # Fallback to raw h5py
        with h5py.File(h5_path, 'r') as f:
            if dataset_name in f:
                data = f[dataset_name][:]
                return pd.DataFrame(data)
            elif 'data' in f:
                data = f['data'][:]
                return pd.DataFrame(data)
            else:
                raise KeyError(f"Dataset {dataset_name} not found in {h5_path}")

def load_h5_data_iterator(h5_path, dataset_name='distances', chunk_size=10000):
    """
    Generator that yields chunks of data from an HDF5 file.
    """
    # Check if it's a pandas store or raw h5py
    try:
        store = pd.HDFStore(h5_path, mode='r')
        if dataset_name not in store and 'data' in store:
            dataset_name = 'data'
        
        n_rows = store.get_storer(dataset_name).nrows
        for start in range(0, n_rows, chunk_size):
            stop = min(start + chunk_size, n_rows)
            yield store.select(dataset_name, start=start, stop=stop)
        store.close()
    except:
        # Fallback to raw h5py chunking
        with h5py.File(h5_path, 'r') as f:
            if dataset_name not in f and 'data' in f:
                dataset_name = 'data'
            
            ds = f[dataset_name]
            for start in range(0, ds.shape[0], chunk_size):
                end = min(start + chunk_size, ds.shape[0])
                yield pd.DataFrame(ds[start:end])


def save_h5_data(data, h5_path, dataset_name='distances', mode='w', format='table', times=None):
    """
    Save data to HDF5 file.
    
    Args:
        data (pd.DataFrame or np.ndarray): Data to save.
        h5_path (str): Path to output .h5 file.
        dataset_name (str): Name for dataset within .h5 file.
        mode (str): 'w' to overwrite, 'a' to append.
        format (str): 'fixed' (faster, no query) or 'table' (slower, queryable).
        times (np.ndarray): Optional time data to save along with distances.
    """
    if isinstance(data, np.ndarray):
        data = pd.DataFrame(data)
    
    data.to_hdf(h5_path, dataset_name, mode=mode, format=format, complevel=9, complib='blosc')
    
    if times is not None:
        with h5py.File(h5_path, 'a') as f:
            if 'times' in f:
                del f['times']
            f.create_dataset('times', data=times, compression="gzip")
            
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
