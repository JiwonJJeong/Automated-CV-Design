import os
import numpy as np
import pandas as pd
from pathlib import Path
import h5py
from typing import Iterator, Dict, List, Optional, Union, Tuple

# =============================================================================
# CONFIGURATION AND CONSTANTS
# =============================================================================

# BASE_DIR should be set to the base_directory_of_analysis mentioned in NEW_README.md
# It can be overriden by the DATA_BASE_DIR environment variable.
DATA_BASE_DIR = "/work/hdd/bfri/jjeong7/analysis_output/dist_maps"
BASE_DIR = os.environ.get("DATA_BASE_DIR", DATA_BASE_DIR)

# Default residues based on the canonical mapping - will be loaded from data files
DEFAULT_RESIDUE_LIST = list(range(144))  # Updated to 144 based on actual data

# Standardized metadata columns that should be ignored by feature selection
METADATA_COLS = {'construct', 'subconstruct', 'replica', 'frame_number', 'time'}

# =============================================================================
# FEATURE NAME GENERATION
# =============================================================================

def load_canonical_residues(data_dir: str = BASE_DIR) -> np.ndarray:
    """
    Loads canonical residue IDs from the canonical_resids.npy file.
    
    Args:
        data_dir (str): Base directory containing the data files.
        
    Returns:
        np.ndarray: Array of canonical residue IDs.
        
    Raises:
        FileNotFoundError: If canonical_resids.npy file is not found.
    """
    # Look for canonical_resids.npy files
    base_path = Path(data_dir)
    
    # Search for canonical_resids.npy files
    resids_files = list(base_path.glob("**/canonical_resids.npy"))
    
    if not resids_files:
        raise FileNotFoundError(f"canonical_resids.npy not found in {data_dir} or its subdirectories")
    
    # Use the first one found (they should be identical across subconstructs)
    canonical_resids = np.load(resids_files[0])
    
    # Only print once per session
    if not hasattr(load_canonical_residues, '_printed'):
        print(f"Loaded {len(canonical_resids)} canonical residues from {resids_files[0]}")
        load_canonical_residues._printed = True
    
    return canonical_resids

def get_feature_cols(df: pd.DataFrame) -> List[str]:
    """
    Returns only the RES_<i>_<j> columns from a DataFrame.
    
    Args:
        df (pd.DataFrame): DataFrame containing feature and metadata columns.
        
    Returns:
        List[str]: List of feature column names (excluding metadata).
    """
    return [c for c in df.columns if c not in METADATA_COLS]

def get_residue_feature_names(residue_list: Optional[List[int]] = None, 
                            data_dir: str = BASE_DIR) -> List[str]:
    """
    Generates pairwise feature names from a list of residue IDs (slots).
    Matches the condensed upper triangle order (i < j) used in the distance matrix.
    Total features: n*(n-1)/2 where n is the number of residues.
    
    Args:
        residue_list (List[int], optional): List of canonical residue slots. 
                                          If None, loads from canonical_resids.npy.
        data_dir (str): Base directory containing the data files.
        
    Returns:
        List[str]: List of strings in 'RES<i>_<j>' format with 1-based indexing.
    """
    if residue_list is None:
        # Load canonical residues from data files
        residue_list = load_canonical_residues(data_dir).tolist()
    
    names = []
    n_res = len(residue_list)
    # The condensed upper triangle order in row-major corresponds to squareform
    for i in range(n_res):
        for j in range(i + 1, n_res):
            # Use 1-based indexing to match PDB residue numbering
            names.append(f"RES{residue_list[i] + 1}_{residue_list[j] + 1}")
    return names

# =============================================================================
# DATA DISCOVERY AND FILE MANAGEMENT
# =============================================================================

def get_data_files(base_dir: str = BASE_DIR) -> List[Dict[str, str]]:
    """
    Discovers all pairwise_dist.h5 and pairwise_dist.npy files and extracts metadata.
    Prefers .h5 if both exist for the same replica.
    
    Expected structure:
    base_dir/
    └── construct/
        └── subconstruct/
            └── {replica}_s{start}_e{end}_pairwise_dist.{h5|npy}
    
    Args:
        base_dir (str): Base directory containing the data files.
        
    Returns:
        List[Dict]: List of file information dictionaries sorted by construct, subconstruct, replica.
    """
    data_files_dict = {}  # key: (construct, subconstruct, replica_str)
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
                    "type": data_file.suffix[1:]  # 'h5' or 'npy'
                }
        except ValueError:
            continue
            
    data_files = list(data_files_dict.values())
    # Sort files by (construct, subconstruct, replica_str)
    data_files.sort(key=lambda x: (x["construct"], x["subconstruct"], x["replica"]))
    return data_files

def get_h5_info(h5_path: str) -> Dict[str, Dict[str, Union[Tuple, str, float]]]:
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

# =============================================================================
# DATA FORMAT CONVERSION
# =============================================================================

def convert_npy_to_h5(npy_path: str, h5_path: str, time_path: Optional[str] = None, 
                     dataset_name: str = 'distances') -> None:
    """
    Converts a legacy .npy distance file to the new HDF5 format.
    
    Args:
        npy_path (str): Path to input .npy file.
        h5_path (str): Path to output .h5 file.
        time_path (str, optional): Path to time.npy file. If None, looks for time.npy in same directory.
        dataset_name (str): Name for the dataset in the H5 file.
    """
    print(f"Converting {npy_path} -> {h5_path}...")
    distances = np.load(npy_path)
    
    # Try to load companion time data
    times = _load_time_data(npy_path, time_path)
    
    with h5py.File(h5_path, 'w') as f:
        f.create_dataset(dataset_name, data=distances, compression="gzip", chunks=True)
        if times is not None:
            # If time.npy is a master file, slice it to match distances
            if len(times) >= len(distances):
                sliced_times = _slice_times_from_filename(npy_path, times, len(distances))
                f.create_dataset('times', data=sliced_times, compression="gzip")
            else:
                f.create_dataset('times', data=times, compression="gzip")
    print("Conversion complete.")

def _load_time_data(npy_path: str, time_path: Optional[str] = None) -> Optional[np.ndarray]:
    """Helper function to load time data from various sources."""
    if time_path and os.path.exists(time_path):
        return np.load(time_path)
    
    # Check in the same directory for time.npy
    potential_time_npy = os.path.join(os.path.dirname(npy_path), 'time.npy')
    if os.path.exists(potential_time_npy):
        return np.load(potential_time_npy)
    
    return None

def _slice_times_from_filename(npy_path: str, times: np.ndarray, target_length: int) -> np.ndarray:
    """Helper function to slice times based on filename patterns."""
    try:
        parts = os.path.basename(npy_path).split('_')
        if len(parts) > 2 and parts[1].startswith('s') and parts[2].startswith('e'):
            start_idx = int(parts[1][1:]) - 1
            end_idx = int(parts[2][1:])
            return times[start_idx:end_idx]
        else:
            return times[:target_length]
    except:
        return times[:target_length]

# =============================================================================
# H5 FILE OPERATIONS
# =============================================================================

def load_h5_data(h5_path: str, dataset_name: str = 'distances', 
                chunk_size: Optional[int] = None) -> Union[pd.DataFrame, Iterator[pd.DataFrame]]:
    """
    Load data from HDF5 file with optional chunking.
    Supports both pandas.to_hdf and raw h5py datasets.
    
    Args:
        h5_path (str): Path to H5 file.
        dataset_name (str): Name of dataset to load.
        chunk_size (int, optional): If provided, returns iterator with chunks of this size.
        
    Returns:
        pd.DataFrame or Iterator[pd.DataFrame]: Data as DataFrame or iterator of DataFrames.
    """
    if chunk_size is not None:
        return load_h5_data_iterator(h5_path, dataset_name, chunk_size)
    
    try:
        # Try pandas first
        return pd.read_hdf(h5_path, dataset_name)
    except:
        # Fallback to raw h5py
        with h5py.File(h5_path, 'r') as f:
            if dataset_name not in f:
                # Fallback to 'data' if 'distances' isn't there
                actual_ds = 'data' if 'data' in f else None
                if not actual_ds:
                    print(f"Warning: No valid dataset found in {h5_path}")
                    return None
            else:
                actual_ds = dataset_name
            
            dataset = f[actual_ds]
            data = dataset[:]
            return pd.DataFrame(data)

def load_h5_data_iterator(h5_path: str, dataset_name: str = 'distances', 
                         chunk_size: int = 10000) -> Iterator[pd.DataFrame]:
    """
    Generator that yields chunks of data from an HDF5 file.
    
    Args:
        h5_path (str): Path to H5 file.
        dataset_name (str): Name of dataset to load.
        chunk_size (int): Size of each chunk.
        
    Yields:
        pd.DataFrame: Chunk of data.
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

def save_h5_data(data: Union[pd.DataFrame, np.ndarray], h5_path: str, 
                dataset_name: str = 'distances', mode: str = 'w', 
                format: str = 'table', times: Optional[np.ndarray] = None) -> None:
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

# =============================================================================
# UNIFIED DATA ITERATORS (H5 + NPY)
# =============================================================================

def _filter_files_by_construct_subconstruct(data_files: List[Dict[str, str]], 
                                          constructs: Optional[List[str]] = None,
                                          subconstructs: Optional[List[str]] = None) -> List[Dict[str, str]]:
    """
    Filter data files by construct and subconstruct criteria.
    
    Args:
        data_files: List of file information dictionaries
        constructs: List of construct names to include (None = all)
        subconstructs: List of subconstruct names to include (None = all)
        
    Returns:
        Filtered list of file information dictionaries
    """
    filtered_files = []
    
    for file_info in data_files:
        # Use metadata from get_data_files (directory structure) instead of filename parsing
        file_construct = file_info.get("construct")
        file_subconstruct = file_info.get("subconstruct")
        
        if not file_construct or not file_subconstruct:
            continue
            
        # Check construct filter
        construct_match = (constructs is None) or (file_construct in constructs)
        
        # Check subconstruct filter  
        subconstruct_match = (subconstructs is None) or (file_subconstruct in subconstructs)
        
        if construct_match and subconstruct_match:
            filtered_files.append(file_info)
            # print(f"✅ Including: {file_info['path']} ({file_construct}/{file_subconstruct})")
    
    print(f"📊 Filtered {len(data_files)} files down to {len(filtered_files)} matching criteria")
    return filtered_files

def data_iterator(base_dir: str = BASE_DIR, chunk_size: int = 10000, 
                 dataset_name: str = 'distances', 
                 constructs: Optional[List[str]] = None,
                 subconstructs: Optional[List[str]] = None) -> Iterator[pd.DataFrame]:
    """
    Unified data iterator that yields DataFrames from both H5 and NPY files.
    Automatically handles the mixed file structure and provides consistent output format.
    
    Args:
        base_dir (str): Base directory containing the data files.
        chunk_size (int): Number of rows per chunk.
        dataset_name (str): Name of dataset to load from H5 files.
        constructs (List[str], optional): Filter by specific constructs only.
        subconstructs (List[str], optional): Filter by specific subconstructs only.
        
    Yields:
        pd.DataFrame: Data chunk with consistent column structure.
    """
    data_files = get_data_files(base_dir)
    
    # Filter files by construct and subconstruct if specified
    if constructs is not None or subconstructs is not None:
        data_files = _filter_files_by_construct_subconstruct(data_files, constructs, subconstructs)
        
        if not data_files:
            print(f"⚠️  No files found matching the specified construct/subconstruct filters:")
            if constructs:
                print(f"   Constructs: {constructs}")
            if subconstructs:
                print(f"   Subconstructs: {subconstructs}")
            return
    
    # Initialize error counter
    if not hasattr(data_iterator, '_error_count'):
        data_iterator._error_count = 0
    
    for file_info in data_files:
        try:
            if file_info["type"] == "h5":
                yield from _iterate_h5_file(file_info, chunk_size, dataset_name)
            elif file_info["type"] == "npy":
                yield from _iterate_npy_file(file_info, chunk_size)
                    
        except Exception as e:
            data_iterator._error_count += 1
            # Only show first 3 error messages
            if data_iterator._error_count <= 3:
                print(f"Error processing {file_info['path']}: {e}")
            elif data_iterator._error_count == 4:
                print("... (suppressing further error messages)")
            continue

def _iterate_h5_file(file_info: Dict[str, str], chunk_size: int, dataset_name: str) -> Iterator[pd.DataFrame]:
    """Helper function to iterate over H5 files."""
    with h5py.File(file_info["path"], 'r') as f:
        # Determine the actual dataset name
        actual_ds = _get_dataset_name(f, dataset_name)
        if actual_ds is None:
            # Only show first warning
            if not hasattr(_iterate_h5_file, '_warning_shown'):
                print(f"Warning: No valid dataset found in some files")
                _iterate_h5_file._warning_shown = True
            return
        
        dataset = f[actual_ds]
        total_rows = dataset.shape[0]
        times = f['times'][:] if 'times' in f else None
        
        # Get column names from dataset attributes or generate from canonical residues
        column_names = dataset.attrs.get('column_names')
        if column_names is None:
            # Generate feature names from canonical residues
            try:
                canonical_resids = load_canonical_residues(os.path.dirname(file_info["path"]))
                column_names = get_residue_feature_names(canonical_resids.tolist())
            except FileNotFoundError:
                # Fallback to generic feature names
                column_names = [f'feature_{i}' for i in range(dataset.shape[1])]
        
        for start_idx in range(0, total_rows, chunk_size):
            end_idx = min(start_idx + chunk_size, total_rows)
            chunk_data = dataset[start_idx:end_idx]
            df = pd.DataFrame(chunk_data, columns=column_names)
            
            # Add metadata columns
            df = _add_metadata_columns(df, file_info, start_idx, end_idx, times)
            yield df

def _iterate_npy_file(file_info: Dict[str, str], chunk_size: int) -> Iterator[pd.DataFrame]:
    """Helper function to iterate over NPY files."""
    # Map-read the npy for memory efficiency
    data = np.load(file_info["path"], mmap_mode='r')
    total_rows = data.shape[0]
    
    # Get column names from canonical residues or generate generic ones
    try:
        canonical_resids = load_canonical_residues(os.path.dirname(file_info["path"]))
        column_names = get_residue_feature_names(canonical_resids.tolist())
    except FileNotFoundError:
        # Fallback to generic feature names
        column_names = [f'feature_{i}' for i in range(data.shape[1])]
    
    # Load time data if available
    times = _load_time_data_for_npy(file_info["path"], total_rows)

    for start_idx in range(0, total_rows, chunk_size):
        end_idx = min(start_idx + chunk_size, total_rows)
        chunk_data = data[start_idx:end_idx]
        df = pd.DataFrame(chunk_data, columns=column_names)
        
        # Add metadata columns
        df = _add_metadata_columns(df, file_info, start_idx, end_idx, times)
        yield df

def _get_dataset_name(h5_file: h5py.File, dataset_name: str) -> Optional[str]:
    """Helper function to determine the actual dataset name in H5 file."""
    if dataset_name in h5_file:
        return dataset_name
    elif 'data' in h5_file:
        return 'data'
    else:
        return None

def _load_time_data_for_npy(npy_path: str, total_rows: int) -> Optional[np.ndarray]:
    """Helper function to load and slice time data for NPY files."""
    potential_time_npy = os.path.join(os.path.dirname(npy_path), 'time.npy')
    if not os.path.exists(potential_time_npy):
        return None
    
    full_times = np.load(potential_time_npy)
    
    # Try to slice based on filename
    try:
        pts = os.path.basename(npy_path).split('_')
        if len(pts) > 2 and pts[1].startswith('s') and pts[2].startswith('e'):
            start_idx = int(pts[1][1:]) - 1
            end_idx = int(pts[2][1:])
            return full_times[start_idx:end_idx]
        else:
            return full_times[:total_rows]
    except:
        return full_times[:total_rows]

def _add_metadata_columns(df: pd.DataFrame, file_info: Dict[str, str], 
                         start_idx: int, end_idx: int, 
                         times: Optional[np.ndarray]) -> pd.DataFrame:
    """Helper function to add metadata columns to DataFrame."""
    df['construct'] = file_info['construct']
    df['subconstruct'] = file_info['subconstruct']
    df['replica'] = file_info['replica']
    
    if times is not None:
        # Ensure the times slice matches the DataFrame length
        time_slice = times[start_idx:end_idx]
        if len(time_slice) == len(df):
            df['time'] = time_slice
        else:
            # If lengths don't match, create a sequential time array
            # Only show this warning once
            if not hasattr(_add_metadata_columns, '_time_warning_shown'):
                print(f"Warning: Time array length mismatch detected. Using sequential time values.")
                _add_metadata_columns._time_warning_shown = True
            df['time'] = np.arange(len(df))
    
    df['frame_number'] = np.arange(start_idx, end_idx) + 1
    return df

# =============================================================================
# LEGACY COMPATIBILITY FUNCTIONS
# =============================================================================

def list_available_constructs_subconstructs(base_dir: str = BASE_DIR) -> Tuple[Dict[str, List[str]], Dict[str, List[str]]]:
    """
    Scan data directory and return available constructs and subconstructs.
    Uses directory structure instead of filename parsing for accuracy.
    
    Args:
        base_dir: Base directory containing data files
        
    Returns:
        Tuple of (constructs_dict, subconstructs_dict) where:
        - constructs_dict: {construct: [subconstructs]}
        - subconstructs_dict: {subconstruct: [constructs]}
    """
    constructs_dict = {}
    subconstructs_dict = {}
    
    base_path = Path(base_dir)
    
    # Scan directory structure: base_dir/construct/subconstruct/
    for construct_dir in base_path.iterdir():
        if not construct_dir.is_dir():
            continue
            
        construct_name = construct_dir.name
        if construct_name not in constructs_dict:
            constructs_dict[construct_name] = []
            
        for subconstruct_dir in construct_dir.iterdir():
            if not subconstruct_dir.is_dir():
                continue
                
            subconstruct_name = subconstruct_dir.name
            
            # Check if this subconstruct has data files
            data_files = list(subconstruct_dir.glob("*_pairwise_dist.*"))
            if not data_files:
                continue
                
            # Build constructs dict
            if subconstruct_name not in constructs_dict[construct_name]:
                constructs_dict[construct_name].append(subconstruct_name)
            
            # Build subconstructs dict
            if subconstruct_name not in subconstructs_dict:
                subconstructs_dict[subconstruct_name] = []
            if construct_name not in subconstructs_dict[subconstruct_name]:
                subconstructs_dict[subconstruct_name].append(construct_name)
    
    # Sort the lists
    for construct in constructs_dict:
        constructs_dict[construct].sort()
    for subconstruct in subconstructs_dict:
        subconstructs_dict[subconstruct].sort()
    
    return constructs_dict, subconstructs_dict


def create_dataframe_factory(base_dir: str = BASE_DIR, 
                             constructs: Optional[List[str]] = None,
                             subconstructs: Optional[List[str]] = None,
                             apply_boundary_filter: bool = True, # Added Toggle
                             n_edge: int = 3,
                             **kwargs):
    
    def factory():
        iterator = data_iterator(base_dir=base_dir, 
                                 constructs=constructs, 
                                 subconstructs=subconstructs, 
                                 **kwargs)
        
        for chunk in iterator:
            if apply_boundary_filter:
                chunk = filter_residue_boundaries(chunk, n_edge=n_edge)
            yield chunk
    
    return factory

# =============================================================================
# MAIN EXECUTION AND TESTING
# =============================================================================

if __name__ == "__main__":
    # Quick test/validation script
    print(f"Searching in: {BASE_DIR}")
    files = get_data_files()
    print(f"Found {len(files)} data files.")
    
    # Test residue loading
    try:
        canonical_resids = load_canonical_residues()
        print(f"Loaded {len(canonical_resids)} canonical residues")
        print(f"Residue range: {canonical_resids[0]} to {canonical_resids[-1]}")
        
        # Test feature name generation with 1-based indexing
        feature_names = get_residue_feature_names()
        print(f"Generated {len(feature_names)} feature names")
        print(f"First 5 features: {feature_names[:5]}")
        print(f"Expected features: {len(canonical_resids) * (len(canonical_resids) - 1) // 2}")
        
        # Test metadata filtering
        print(f"Metadata columns: {sorted(METADATA_COLS)}")
        
    except FileNotFoundError as e:
        print(f"Warning: {e}")
        print("Using fallback feature names.")
    
    if files:
        print("\nFirst 5 files found:")
        for f in files[:5]:
            print(f"  {f['construct']}/{f['subconstruct']} - Replica {f['replica']} -> {os.path.basename(f['path'])}")
        
        print("\nTesting iterator (first chunk):")
        for i, chunk in enumerate(data_iterator(chunk_size=10)):
            print(f"Chunk 0 shape: {chunk.shape}")
            print("Columns:", chunk.columns.tolist()[:5], "...", chunk.columns.tolist()[-5:])
            
            # Test feature column extraction
            feature_cols = get_feature_cols(chunk)
            print(f"Feature columns: {len(feature_cols)}")
            print(f"First 3 feature cols: {feature_cols[:3]}")
            
            print("First row Sample:\n", chunk.iloc[0][:5])
            break
        
        # Test the factory function (exhausted generator fix)
        print("\nTesting DataFrame factory:")
        factory = create_dataframe_factory(chunk_size=5)
        
        # First call
        print("First factory call:")
        chunk1 = next(factory())
        print(f"  Shape: {chunk1.shape}")
        
        # Second call (should work without exhaustion)
        print("Second factory call:")
        chunk2 = next(factory())
        print(f"  Shape: {chunk2.shape}")
        print("✓ Factory successfully creates fresh iterators!")

import re

def filter_residue_boundaries(df, n_edge=3):
    # Updated regex to match 'RES1_2' format from your get_residue_feature_names()
    res_pattern = re.compile(r'RES(\d+)_(\d+)', re.IGNORECASE)
    res_cols = [col for col in df.columns if res_pattern.match(col)]
    
    if not res_cols:
        return df

    # Extract indices
    all_indices = []
    for col in res_cols:
        nums = res_pattern.match(col).groups()
        all_indices.extend([int(nums[0]), int(nums[1])])
    
    unique_indices = sorted(list(set(all_indices)))
    
    # Define forbidden boundary numbers
    low_bounds = set(unique_indices[:n_edge])
    high_bounds = set(unique_indices[-n_edge:])
    forbidden = low_bounds.union(high_bounds)
    
    cols_to_keep = []
    for col in df.columns:
        match = res_pattern.match(col)
        if match:
            idx1, idx2 = map(int, match.groups())
            if idx1 not in forbidden and idx2 not in forbidden:
                cols_to_keep.append(col)
        else:
            cols_to_keep.append(col) # Keep 'class', 'construct', etc.
            
    return df[cols_to_keep]