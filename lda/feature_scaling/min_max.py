import pandas as pd
import numpy as np
from typing import Callable

# Type alias for the generator function
DFGenerator = Callable[[], pd.DataFrame]

def create_minmax_scaled_generator(df_factory: DFGenerator) -> DFGenerator:
    """
    Accepts a dataframe generator, calculates min/max bounds,
    and returns a new generator that yields normalized data.
    """
    # 1. "Fit" phase: Establish the global Min and Max
    reference_df = df_factory()
    numeric_cols = reference_df.select_dtypes(include=[np.number]).columns
    
    col_min = reference_df[numeric_cols].min()
    col_max = reference_df[numeric_cols].max()
    
    # Calculate the range (denominator)
    col_range = col_max - col_min

    # 2. "Transform" phase: Define the new factory
    def normalized_factory() -> pd.DataFrame:
        df = df_factory()
        
        # Apply the scaling: (x - min) / (max - min)
        # We use .clip() to ensure that if new data slightly exceeds 
        # the original bounds, it stays within the [0, 1] range.
        scaled_values = (df[numeric_cols] - col_min) / col_range
        df[numeric_cols] = scaled_values.clip(0, 1)
        
        return df

    return normalized_factory

# --- Example Usage ---
if __name__ == "__main__":
    pass