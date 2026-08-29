"""
Preprocessing utilities for the Loan Default Prediction app.

The trained model expects a one-hot encoded feature vector matching
the exact columns it was trained on (saved in loan_default_columns.pkl).
This module rebuilds that feature vector from raw form inputs.
"""

import pandas as pd


def build_input_dataframe(numeric_inputs: dict, categorical_inputs: dict,
                           categorical_options: dict, model_columns: list) -> pd.DataFrame:
    """
    Reconstructs a single-row DataFrame that matches the model's expected columns.

    numeric_inputs: dict of {column_name: value} for numeric fields
    categorical_inputs: dict of {column_name: selected_value} for categorical fields
    categorical_options: dict of {column_name: [sorted unique values]} (saved during training)
    model_columns: the exact list of columns the model was trained on
    """
    row = {}

    # Numeric fields go in directly
    row.update(numeric_inputs)

    # Recreate one-hot encoding to match training (drop_first=True was used,
    # so the first value in each sorted options list was dropped as baseline)
    for col, selected_value in categorical_inputs.items():
        options = categorical_options[col]
        baseline = options[0]
        for value in options[1:]:
            dummy_col_name = f"{col}_{value}"
            row[dummy_col_name] = 1 if selected_value == value else 0

    df = pd.DataFrame([row])

    # Ensure every column the model expects exists, in the right order.
    # Any column not present (shouldn't normally happen) is filled with 0.
    df = df.reindex(columns=model_columns, fill_value=0)

    return df
