import pandas as pd
import numpy as np
from typing import Callable

# Define a type alias for clarity: a function that returns a DataFrame
DFGenerator = Callable[[], pd.DataFrame]

def create_standard_scaled_generator(df_factory: DFGenerator) -> DFGenerator:
    """
    Accepts a dataframe generator, calculates scaling parameters,
    and returns a new generator that yields scaled data.
    """
    # 1. "Fit" phase: Run the generator once to learn the distribution
    # In a production environment with massive data, you might use 
    # a partial_fit approach, but for a standard factory:
    reference_df = df_factory()
    
    # Identify numeric columns only
    numeric_cols = reference_df.select_dtypes(include=[np.number]).columns
    
    # Calculate means and standard deviations
    means = reference_df[numeric_cols].mean()
    stds = reference_df[numeric_cols].std()

    # 2. "Transform" phase: Define the new factory
    def scaled_factory() -> pd.DataFrame:
        # Get a fresh copy of the data
        df = df_factory()
        
        # Apply the scaling formula: z = (x - u) / s
        # Using LaTeX for the logic: $z = \frac{x - \mu}{\sigma}$
        df[numeric_cols] = (df[numeric_cols] - means) / stds
        
        return df

    return scaled_factory

# --- Example Usage ---
if __name__ == "__main__":
    pass