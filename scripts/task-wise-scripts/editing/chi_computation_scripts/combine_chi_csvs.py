import pandas as pd
import os

def combine_chi_csvs(prefix, output_file=None):
    """
    Combines x_python_chi.csv, x_cpp_chi.csv, and x_java_chi.csv into x_chi.csv
    
    Args:
        prefix: The prefix of the files (e.g., 'dataset', 'claude', 'deepseek')
        output_file: Optional output file name. If None, uses '{prefix}_chi.csv'
    """
    
    # Define input files
    python_file = f"{prefix}_python_chi.csv"
    cpp_file = f"{prefix}_cpp_chi.csv"
    java_file = f"{prefix}_java_chi.csv"
    
    # Define output file
    if output_file is None:
        output_file = f"{prefix}_chi.csv"
    
    print(f"Loading files:")
    print(f"  - {python_file}")
    print(f"  - {cpp_file}")
    print(f"  - {java_file}")
    
    for file in [python_file, cpp_file, java_file]:
        if not os.path.exists(file):
            print(f"Error: {file} not found!")
            return
    
    df_python = pd.read_csv(python_file)
    df_cpp = pd.read_csv(cpp_file)
    df_java = pd.read_csv(java_file)
    
    print(f"\nFile sizes:")
    print(f"  - Python: {len(df_python)} rows")
    print(f"  - C++: {len(df_cpp)} rows")
    print(f"  - Java: {len(df_java)} rows")
    
    combined_df = pd.concat([df_python, df_cpp, df_java], ignore_index=True)
    
    print(f"\nCombined: {len(combined_df)} rows total")
    print(f"Columns: {list(combined_df.columns)}")
    
    combined_df.to_csv(output_file, index=False)
    print(f"\nSuccessfully saved to: {output_file}")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python combine_chi_csvs.py <prefix> [output_file]")
        print("\nExamples:")
        print("  python combine_chi_csvs.py dataset")
        print("  python combine_chi_csvs.py claude claude_all_chi.csv")
        print("  python combine_chi_csvs.py deepseek")
        sys.exit(1)
    
    prefix = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else None
    
    combine_chi_csvs(prefix, output_file)
