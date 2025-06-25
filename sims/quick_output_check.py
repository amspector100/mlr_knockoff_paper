import glob
import pandas as pd
import argparse
import os

def str_to_list(s):
    return [x.strip() for x in s.split(",")]

def check_job_outputs(job_id, group_cols="n,p,method", meas_cols="power,fdp"):
    """
    Check outputs for a given job_id by finding and analyzing CSV files and reporting mean measurements.
    
    Parameters:
    job_id (str): The job ID to search for in file paths
    group_cols (str or list): Comma-separated column names or list of columns to group by
    """
    # Get the directory of the current file and construct path to ../data
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(current_dir, "..", "data")
    
    # Find all CSV files with the job_id recursively in the data/ directory
    pattern = os.path.join(data_dir, "**", f"*{job_id}*.csv")
    csv_files = glob.glob(pattern, recursive=True)
    
    if not csv_files:
        print(f"No CSV files found with job_id '{job_id}' in {data_dir} directory")
        return
    
    print(f"Found {len(csv_files)} CSV files with job_id '{job_id}'")
    
    # Read all CSV files and concatenate them
    dataframes = []
    for file in csv_files:
        try:
            df = pd.read_csv(file)
            dataframes.append(df)
        except Exception as e:
            print(f"Error reading {file}: {e}")
    
    if not dataframes:
        print("No valid CSV files could be read")
        return
    
    # Concatenate all dataframes
    combined_df = pd.concat(dataframes, ignore_index=True)
    
    # Parse group_cols if it's a string
    group_cols = str_to_list(group_cols)
    meas_cols = str_to_list(meas_cols)
    
    # Group by specified columns and calculate mean power, etc
    required_cols = group_cols + meas_cols
    if 'seed' in combined_df.columns:
        print(f"NUMBER OF SEEDS: {len(combined_df['seed'].unique())}.")
    if all(col in combined_df.columns for col in required_cols):
        grouped_means = combined_df.groupby(group_cols)[meas_cols].mean()
        print(f"\nMean {meas_cols} by ({', '.join(group_cols)}):")
        print(grouped_means.reset_index())
    else:
        missing_cols = [col for col in required_cols if col not in combined_df.columns]
        print(f"Missing required columns: {missing_cols}")
        print(f"Available columns: {list(combined_df.columns)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--job_id", type=str, required=True)
    parser.add_argument("--group_cols", type=str, default="n,p,method")
    parser.add_argument("--meas_cols", type=str, default="power")
    args = parser.parse_args()
    check_job_outputs(job_id=args.job_id, group_cols=args.group_cols, meas_cols=args.meas_cols)

if __name__ == "__main__":
    main()