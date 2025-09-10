"""
utils.py

Purpose
-------
    Provide utility functions for data loading, preprocessing, and basic 
    feature engineering to support model training.

Description
-----------
    This module includes:
    1. load_csv_data: Safely load data from CSV into pandas DataFrame.
    2. train_test_split: Split dataset into training and test sets.
    3. calculate_bmi: Compute BMI from height and weight.
    4. data_encoding: Encode categorical values into numerical form 
       (activity, diet, family history).
    5. label_encoding: Dictionary mapping categorical values to integers.

Usage
-----
    These utilities are used by the training script to prepare and preprocess 
    the dataset before feeding it into the Random Forest model.

Dependencies
------------
    - pandas
    - numpy
    - pathlib
    - typing

Author
------
    Waseem Alyazidi.

Date
----
    2025-09-09.
"""


import pandas as pd # type: ignore
import numpy as np
from pathlib import Path
from typing import Optional, Union, Tuple, cast, Dict, Union


# Load data from CSV file.
def load_csv_data(source_path: Union[str, Path], encoding: str = "utf-8") -> pd.DataFrame:
    """
        Load data from a CSV file into a pandas DataFrame.

        Parameters:
            source_path (str, Path): Path to the CSV file.
            encoding (str): Encoding to use when reading the CSV file. (Default = `utf-8`).

        Returns:
            pd.DataFrame: Loaded data as a DataFrame.

        Raises:
            FileNotFoundError: If the file does not exist or is not a CSV file.
            ValueError: If the file path does not have a .csv extension.
            IOError: If an error occurs while reading the CSV file.
    """
    path = Path(source_path)

    # Invalid paths
    if not path.exists(): # File not exists
        raise FileNotFoundError(f"The file `{source_path}` does not exist.\n")
    if path.suffix.lower() != ".csv": # Invalid extension
        raise ValueError(f"The file `{source_path}` is not a CSV file.\n")
    
    try:
        df: pd.DataFrame = pd.read_csv(source_path, encoding=encoding)
    except Exception as e:
        raise IOError(f"Error loading CSV data from `{source_path}`:\n{e}")
    return df

# Train\Test split
def train_test_split(df: pd.DataFrame, test_size: float = 0.2,
                     target_col: str = "target", random_state: Optional[int] = 42) -> Tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
        Split data into X_train, y_train, X_test, y_test.

        Parameters:
            df (pd.DataFrame): Input DataFrame with features and target variable.
                The last column is assumed to be the target variable.
            test_size (float): Proportion of the dataset to include in the test split. (Default = 0.2).
            target_col (str): Target colomn name. (Default = `target`)
            random_state (int, optional): Random seed for reproducibility. (Default = 42).

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: 
                X_train, y_train, X_test, y_test arrays.
        
        Raises:
            ValueError: If test_split is not between 0 and 1, or if DataFrame has fewer than 2 rows.
    """
    if not 0 < test_size < 1:
        raise ValueError(f"test_split must be a float between 0 and 1. Got {test_size}.\n")
    if df.shape[0] < 2:
        raise ValueError(f"DataFrame must have at least 2 rows. Got df.shape[0]: {df.shape[0]}.\n")
    
    n_samples: int = df.shape[0]

    if isinstance(target_col, str):
        target_col_index: int = cast(int, df.columns.get_loc(target_col))

    data: np.ndarray = df.to_numpy()
    # Split to X and y
    X: np.ndarray = data[:, np.arange(data.shape[1]) != target_col_index]
    y: np.ndarray = data[:, target_col_index]

    # Shuffle data
    rng: np.random.Generator = np.random.default_rng(random_state)
    indices: np.ndarray = rng.permutation(n_samples)

    test_split_size: int = int(n_samples * test_size)

    # Split train\test
    train_indices: np.ndarray = indices[test_split_size:]
    test_indices: np.ndarray = indices[:test_split_size]

    return X[train_indices], y[train_indices], X[test_indices], y[test_indices]


# Calculate bmi
def calculate_bmi(height: float, weight: float) -> float:
    """
        Calculate BMI.

        Parameters:
            height (float): cm height.
            weight (float): kg weight.

        Returns:
            float: The BMI.
    """
    if height <= 0 or weight <= 0:
        raise ValueError(f"Invalid input. height and weight can not be zero or negative.")
    
    m_height: float = height/100
    return weight/(m_height**2)


label_encoding: Dict[str, Dict[str, int]] = {
    "activity": {"low": 0, "normal": 1, "high": 2},
    "diet": {"healthy": 0, "normal": 1, "unhealthy": 2},
    "family_history": {"no": 0, "yes": 1}
}

# Encoding categorical data
def data_encoding(col_name: str, value: str) -> int:
    """
        Encoding categorical data.

        Parameters:
            col_name (str): Colomn name (e.g. `activity`).
            value (str): String value (e.g. `low`).

        Returns:
            int: Encoding value.
    """
    if col_name in label_encoding and value in label_encoding[col_name]:
        encoded_value: int = label_encoding[col_name][value]
    else:
        encoded_value = -1
    return encoded_value