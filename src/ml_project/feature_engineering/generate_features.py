"""
Feature Generation Module.
This module creates interaction terms and composite variables 
to improve the predictive performance of the PM2.5 regression model, and 
creates sophisticated cyclical datetime transformations for time-series air quality data.
"""

import numpy as np
import pandas as pd

def generate_interaction_features(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Creates composite features to capture relationships between weather variables.
    (Kept as per previous implementation).
    """
    # Create a copy to prevent modifying the original dataframe
    df_enhanced = dataframe.copy()

    # 1. Weather Interaction: Temperature * Wind Speed
    # Rationale: Higher wind speeds usually help disperse pollutants, 
    # but the effect can vary depending on the temperature.
    if 'tavg' in df_enhanced.columns and 'wspd' in df_enhanced.columns:
        df_enhanced['temp_wind_interaction'] = (
            df_enhanced['tavg'] * df_enhanced['wspd']
        )

    # 2. Atmospheric Stability: Temperature Range
    # Rationale: A large difference between max and min temperature 
    # can indicate air stagnation, which traps PM2.5.
    if 'tmax' in df_enhanced.columns and 'tmin' in df_enhanced.columns:
        df_enhanced['temp_diurnal_range'] = (
            df_enhanced['tmax'] - df_enhanced['tmin']
        )

    # 3. Combustion Indicator: NO2 / CO Ratio
    # Rationale: Different ratios can point to traffic vs. industrial sources.
    if 'no2' in df_enhanced.columns and 'co' in df_enhanced.columns:
        # Added 0.001 to prevent division by zero
        df_enhanced['pollutant_ratio'] = (
            df_enhanced['no2'] / (df_enhanced['co'] + 0.001)
        )

    return df_enhanced

def generate_cyclical_date_features(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Transforms the 'reading_date' into cyclical Sine and Cosine features.
    This allows the model to understand that the end of the year/month 
    is chronologically close to the beginning.
    """
    df_cyclical = dataframe.copy()

    if 'reading_date' in df_cyclical.columns:
        # Convert to datetime object
        df_cyclical['reading_date'] = pd.to_datetime(df_cyclical['reading_date'])
        
        # Extract numerical time components
        day_of_year = df_cyclical['reading_date'].dt.dayofyear
        month = df_cyclical['reading_date'].dt.month
        
        # Day of Year Cycle (using 365.25 for leap years)
        df_cyclical['day_of_year_sin'] = np.sin(2 * np.pi * day_of_year / 365.25)
        df_cyclical['day_of_year_cos'] = np.cos(2 * np.pi * day_of_year / 365.25)
        
        # Monthly Cycle
        df_cyclical['month_sin'] = np.sin(2 * np.pi * month / 12)
        df_cyclical['month_cos'] = np.cos(2 * np.pi * month / 12)

    return df_cyclical

def test_feature_logic(dataframe: pd.DataFrame) -> None:
    """
    Unit test to verify that both interaction and cyclical features 
    exist and are mathematically valid.
    """
    # Check Interaction Features
    assert 'temp_wind_interaction' in dataframe.columns, "Interaction features missing!"
    
    # Check Cyclical Features
    if 'day_of_year_sin' in dataframe.columns:
        assert dataframe['day_of_year_sin'].between(-1.01, 1.01).all(), "Sine out of bounds!"
        print("Unit Test Passed: All features are present and valid.")



"""
Data Splitting Module.

This module is responsible for dividing the preprocessed and scaled dataset 
into training and testing subsets. This ensures that the Machine Learning 
model can be trained on one subset and objectively evaluated on another.
"""

from typing import Tuple
import pandas as pd
from sklearn.model_selection import train_test_split

def split_dataset(
    dataframe: pd.DataFrame, 
    target_col: str = 'pm25', 
    test_size: float = 0.2, 
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Separates the target variable from the features and splits the data 
    into an 80/20 training and testing set.
    
    Args:
        dataframe (pd.DataFrame): The fully processed dataset.
        target_col (str): The column representing the target variable to predict.
        test_size (float): The proportion of the dataset to allocate to the test set.
        random_state (int): A seed value to ensure the split is reproducible.
        
    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]: 
            A tuple containing (X_train, X_test, y_train, y_test).
            
    Raises:
        ValueError: If the target column is missing from the dataframe.
    """
    if target_col not in dataframe.columns:
        raise ValueError(f"Error: Target column '{target_col}' not found in the dataset.")

    # Isolate the independent variables (Features) and dependent variable (Target)
    x_features = dataframe.drop(columns=[target_col])
    y_target = dataframe[target_col]

    # Perform the split
    x_train, x_test, y_train, y_test = train_test_split(
        x_features, 
        y_target, 
        test_size=test_size, 
        random_state=random_state
    )

    return x_train, x_test, y_train, y_test


def test_data_split(
    x_train: pd.DataFrame, 
    x_test: pd.DataFrame, 
    y_train: pd.Series, 
    y_test: pd.Series
) -> None:
    """
    REQUIREMENT: Unit testing function.
    Validates that the dimensions of the generated training and testing 
    sets align perfectly to prevent model errors.
    """
    # Verify that the number of rows in features matches the target labels
    assert x_train.shape[0] == y_train.shape[0], "Training features and labels length mismatch!"
    assert x_test.shape[0] == y_test.shape[0], "Testing features and labels length mismatch!"
    
    # Verify that the number of columns (features) is identical in both sets
    assert x_train.shape[1] == x_test.shape[1], "Mismatch in number of features between train and test sets!"
    
    print("Unit Test Passed: Train-Test Split executed successfully.")
    print(f"-> Training Set Size:   {x_train.shape[0]} rows")
    print(f"-> Testing Set Size:    {x_test.shape[0]} rows")
    print(f"-> Number of Features:  {x_train.shape[1]} columns")