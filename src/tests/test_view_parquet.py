#!/usr/bin/env python3
# filename: test_view_parquet.py

import pandas as pd
import sys

def view_parquet(file_path, n=5):
    """
    View the content of a Parquet file

    Parameters:
    file_path: path to the Parquet file
    n: number of rows to display (default 5)
    """
    try:
        # Read Parquet file
        df = pd.read_parquet(file_path)

        # Show full column content
        pd.set_option('display.max_columns', None)   # show all columns
        pd.set_option('display.max_rows', None)      # show all rows
        pd.set_option('display.max_colwidth', None)  # show full content of each cell

        # Print basic information
        print("=== File Info ===")
        print(f"Rows: {len(df)}, Columns: {len(df.columns)}")
        print("\nColumn names and data types:")
        print(df.dtypes)

        # Print first n rows
        print(f"\n=== First {n} Rows ===")
        print_df = df.copy()
        for col in print_df.columns:
            if print_df[col].dtype == 'object':
                print_df[col] = print_df[col].apply(lambda x: len(x) if isinstance(x, (list, dict)) else x)
        print(print_df.head(n))

    except Exception as e:
        print(f"Error reading file: {e}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python test_view_parquet.py <file_path> [n]")
        sys.exit(1)

    file_path = sys.argv[1]
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 5

    view_parquet(file_path, n)
