import pandas as pd
from typing import List, Optional
import csv

def find_missing_rows(
    file1_path: str,
    file2_path: str,
    column: str,
    output_path: str,
    columns_to_compare: Optional[List[str]] = None
) -> None:
    """
    Compare two CSV files based on a specific column and create a new CSV file
    containing rows that exist in the first file but not in the second file.
    
    Args:
        file1_path (str): Path to the first CSV file (source)
        file2_path (str): Path to the second CSV file (target)
        column (str): Column name to compare
        output_path (str): Path where the output CSV will be saved
        columns_to_compare (List[str], optional): Specific columns to include in the output.
            If None, all columns from file1 will be included.
    
    Returns:
        None: Creates a new CSV file with the missing rows
    """
    try:
        # Read both CSV files
        df1 = pd.read_csv(file1_path)
        df2 = pd.read_csv(file2_path)
        
        # Ensure the column exists in both files
        if column not in df1.columns or column not in df2.columns:
            raise ValueError(f"Column '{column}' not found in one or both files")
        
        # Get unique values from the comparison column in both files
        values_in_file1 = set(df1[column].unique())
        values_in_file2 = set(df2[column].unique())
        
        # Find values that are in file1 but not in file2
        missing_values = values_in_file1 - values_in_file2
        
        # Filter rows from file1 that have missing values
        missing_rows = df1[df1[column].isin(missing_values)]
        
        # If specific columns are requested, filter them
        if columns_to_compare:
            # Ensure all requested columns exist in file1
            missing_columns = [col for col in columns_to_compare if col not in df1.columns]
            if missing_columns:
                raise ValueError(f"Columns not found in file1: {missing_columns}")
            missing_rows = missing_rows[columns_to_compare]
        
        # Save the result to a new CSV file
        missing_rows.to_csv(output_path, index=False)
        
        # Print summary
        print(f"Total rows in file1: {len(df1)}")
        print(f"Total rows in file2: {len(df2)}")
        print(f"Number of missing rows: {len(missing_rows)}")
        print(f"Results saved to: {output_path}")
        
    except Exception as e:
        print(f"Error: {str(e)}")
        raise

if __name__ == "__main__":
    # find_missing_rows(
    #     file2_path="official_repos_mcp_analysis.csv",
    #     file1_path="official_repos_with_metadata.csv",
    #     column="repo",  # or any other column you want to compare
    #     output_path="missing.csv",
    # )

    df = pd.read_csv('mcp_analysis.csv')

    # Calculate the overall score
    df['overall_score'] = (
        df['doc_quality_score'] /10 * 0.30 +
        df['security_score'] * 0.30 +
        df['popularity_score']/5 * 0.40
    )

    # Write the updated DataFrame to a new CSV file
    df.to_csv('official_repos_mcp_analysis_with_overall_score.csv', index=False)

    print("New CSV file created: official_repos_mcp_analysis_with_overall_score.csv")
    print("Average Overall score:" + str(df['overall_score'].mean()))


