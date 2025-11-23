"""
FET Data Processor Module
Contains all functions for loading, merging, processing, and plotting FET measurement data.
"""

import pandas as pd
import numpy as np
import os
from pathlib import Path
import re
import matplotlib.pyplot as plt


def extract_fet_id(filename):
    """
    Extract FET ID from filename.
    Format: 'FD-x_cx_Dx-x'
    Example: 'FD-2_c2_D3-1_n_comment.csv' -> 'FD-2_c2_D3-1'
    """
    match = re.search(r'FD-\d+_c\d+_D\d+-\d+', filename)
    if match:
        return match.group(0)
    return None


def extract_sweep_type(filename):
    """
    Extract sweep type ('n' or 'p') from filename.
    Looks for '_n_' or '_p_' after the device ID.
    Example: 'FD-2_c2_D3-1_n_comment.csv' -> 'n'
    """
    match = re.search(r'FD-\d+_c\d+_D\d+-\d+_([np])', filename)
    if match:
        return match.group(1)
    return None


def load_and_group_data(data_directory):
    """
    Load all CSV files and group them by FET ID.
    Skips files starting with 'noConn' or 'Leak'.
    Groups multiple files (n and p sweeps) for each FET.
    
    Args:
        data_directory: Path to folder containing CSV files
        
    Returns:
        Dictionary with FET IDs as keys and dict of dataframes as values
        Each FET has {'n': df_negative, 'p': df_positive, 'all': [list of all dfs]}
    """
    data_dir = Path(data_directory)
    
    # Dictionary to store data grouped by FET ID
    fet_groups = {}
    
    # Counters for dead FETs
    noconn_count = 0
    leak_count = 0
    
    # Find all CSV files
    csv_files = list(data_dir.glob('*.csv'))
    print(f"Found {len(csv_files)} CSV files\n")
    
    # Process each file
    for file_path in csv_files:
        filename = file_path.name
        
        # Check if file is a dead FET (starts with noConn or Leak)
        if filename.startswith('noConn'):
            noconn_count += 1
            print(f"Skipped (noConn): {filename}")
            continue
        elif filename.startswith('Leak'):
            leak_count += 1
            print(f"Skipped (Leak): {filename}")
            continue
        
        # Extract FET ID from filename
        fet_id = extract_fet_id(filename)
        sweep_type = extract_sweep_type(filename)
        
        if fet_id:
            # Read the CSV file
            df = pd.read_csv(file_path)
            
            # Default to positive sweep if not specified
            if sweep_type is None:
                sweep_type = 'p'
            
            # Add metadata columns
            df['source_file'] = filename
            df['sweep_type'] = sweep_type
            
            # Initialize FET group if needed
            if fet_id not in fet_groups:
                fet_groups[fet_id] = {'n': None, 'p': None, 'all': []}
            
            # Store by sweep type
            if sweep_type == 'n':
                fet_groups[fet_id]['n'] = df
                print(f"Loaded: {filename} -> FET ID: {fet_id} (negative sweep)")
            elif sweep_type == 'p':
                fet_groups[fet_id]['p'] = df
                sweep_label = "positive sweep" if extract_sweep_type(filename) else "positive sweep - default"
                print(f"Loaded: {filename} -> FET ID: {fet_id} ({sweep_label})")
            
            # Add to list of all files for this FET
            fet_groups[fet_id]['all'].append(df)
            
        else:
            print(f"Warning: Could not extract FET ID from {filename}")
    
    # Print summary to console
    print(f"\n{'='*50}")
    print(f"Summary:")
    print(f"  Working FETs: {len(fet_groups)} unique devices")
    print(f"  Dead FETs skipped:")
    print(f"    - No Connection (noConn): {noconn_count}")
    print(f"    - Leakage (Leak): {leak_count}")
    print(f"    - Total dead: {noconn_count + leak_count}")
    print(f"\nWorking FET IDs and their measurements:")
    for fet_id in sorted(fet_groups.keys()):
        n_status = "✓" if fet_groups[fet_id]['n'] is not None else "✗"
        p_status = "✓" if fet_groups[fet_id]['p'] is not None else "✗"
        total_files = len(fet_groups[fet_id]['all'])
        print(f"  {fet_id}: {total_files} file(s) [n:{n_status} p:{p_status}]")
    print(f"{'='*50}\n")
    
    # Write summary to text file
    summary_file = data_dir / "loading_summary.txt"
    with open(summary_file, 'w') as f:
        f.write("="*50 + "\n")
        f.write("FET Data Loading Summary\n")
        f.write("="*50 + "\n\n")
        f.write(f"Total CSV files found: {len(csv_files)}\n")
        f.write(f"Working FETs loaded: {len(fet_groups)} unique devices\n\n")
        f.write("Dead FETs skipped:\n")
        f.write(f"  - No Connection (noConn): {noconn_count}\n")
        f.write(f"  - Leakage (Leak): {leak_count}\n")
        f.write(f"  - Total dead: {noconn_count + leak_count}\n\n")
        f.write("="*50 + "\n")
        f.write("Working FET Details:\n")
        f.write("="*50 + "\n\n")
        
        for fet_id in sorted(fet_groups.keys()):
            n_status = "Yes" if fet_groups[fet_id]['n'] is not None else "No"
            p_status = "Yes" if fet_groups[fet_id]['p'] is not None else "No"
            total_files = len(fet_groups[fet_id]['all'])
            
            f.write(f"FET ID: {fet_id}\n")
            f.write(f"  Total files: {total_files}\n")
            f.write(f"  Negative sweep: {n_status}\n")
            f.write(f"  Positive sweep: {p_status}\n")
            
            # List all files for this FET
            f.write(f"  Files:\n")
            for df in fet_groups[fet_id]['all']:
                f.write(f"    - {df['source_file'].iloc[0]} (sweep: {df['sweep_type'].iloc[0]})\n")
            f.write("\n")
    
    print(f"Summary written to: {summary_file}\n")
    
    return fet_groups


def merge_sweeps(fet_groups):
    """
    Merge n and p sweep dataframes for each FET into a single dataframe.
    Merges by matching column names.
    
    Args:
        fet_groups: Dictionary from load_and_group_data()
        
    Returns:
        Dictionary with FET IDs as keys and merged dataframes as values
    """
    merged_data = {}
    
    print("\n" + "="*50)
    print("Merging sweep data for each FET")
    print("="*50 + "\n")
    
    for fet_id, data in fet_groups.items():
        dfs_to_merge = []
        
        # Collect available dataframes
        if data['n'] is not None:
            dfs_to_merge.append(data['n'])
        if data['p'] is not None:
            dfs_to_merge.append(data['p'])
        
        # Merge dataframes by column names
        if len(dfs_to_merge) > 0:
            # Use concat with axis=0 to stack rows, matching by column names
            merged_df = pd.concat(dfs_to_merge, axis=0, ignore_index=True, sort=False)
            
            # Remove empty columns (all NaN or all empty strings)
            merged_df = merged_df.dropna(axis=1, how='all')
            merged_df = merged_df.loc[:, (merged_df != '').any(axis=0)]
            
            merged_data[fet_id] = merged_df
            
            n_rows = len(data['n']) if data['n'] is not None else 0
            p_rows = len(data['p']) if data['p'] is not None else 0
            
            print(f"{fet_id}:")
            print(f"  Negative sweep: {n_rows} rows")
            print(f"  Positive sweep: {p_rows} rows")
            print(f"  Merged total: {len(merged_df)} rows")
            print(f"  Columns: {list(merged_df.columns)}")
            
            # Check if there are any NaN values from mismatched columns
            if merged_df.isnull().any().any():
                print(f"  Warning: Some columns had mismatched data (NaN values present)")
            
            print()
    
    print(f"{'='*50}")
    print(f"Merged {len(merged_data)} FETs successfully")
    print(f"{'='*50}\n")
    
    return merged_data


def process_fet_data(merged_data):
    """
    Process FET data: sort by Vg, scale currents, and calculate transconductance.
    
    Processing steps:
    1. Sort dataframe by Vg column
    2. Create 'Drain Current (nA)' column = Id * 1e9
    3. Create 'Gate Current (pA)' column = Ig * 1e12
    4. Create 'gm' column = derivative of Id with respect to Vg
    
    Args:
        merged_data: Dictionary of merged dataframes from merge_sweeps()
        
    Returns:
        Dictionary with FET IDs as keys and processed dataframes as values
    """
    processed_data = {}
    
    print("\n" + "="*50)
    print("Processing FET data")
    print("="*50 + "\n")
    
    for fet_id, df in merged_data.items():
        # Make a copy to avoid modifying original data
        processed_df = df.copy()
        
        # Check if required columns exist
        if 'Vg' not in processed_df.columns:
            print(f"{fet_id}: Warning - 'Vg' column not found, skipping")
            continue
        
        # Step 1: Sort by Vg
        processed_df = processed_df.sort_values('Vg').reset_index(drop=True)
        
        # Step 2: Create Drain Current (nA) column
        if 'Id' in processed_df.columns:
            processed_df['Drain Current (nA)'] = processed_df['Id'] * 1e9
        else:
            print(f"{fet_id}: Warning - 'Id' column not found")
        
        # Step 3: Create Gate Current (pA) column
        if 'Ig' in processed_df.columns:
            processed_df['|IG| (pA)'] = abs(processed_df['Ig']) * 1e12
        else:
            print(f"{fet_id}: Warning - 'Ig' column not found")
        
        # Step 4: Calculate transconductance (gm = dId/dVg)
        if 'Id' in processed_df.columns and 'Vg' in processed_df.columns:
            # Use numpy gradient for numerical derivative
            vg = processed_df['Vg'].values
            id_vals = processed_df['Id'].values
            gm = np.gradient(id_vals, vg)
            processed_df['gm'] = gm
        else:
            print(f"{fet_id}: Warning - Cannot calculate gm, missing 'Id' or 'Vg'")
        
        processed_data[fet_id] = processed_df
        
        print(f"{fet_id}: Processed successfully")
        print(f"  Rows: {len(processed_df)}")
        print(f"  Vg range: {processed_df['Vg'].min():.3f} to {processed_df['Vg'].max():.3f} V")
        if 'gm' in processed_df.columns:
            print(f"  gm range: {processed_df['gm'].min():.3e} to {processed_df['gm'].max():.3e} S")
        print()
    
    print(f"{'='*50}")
    print(f"Processed {len(processed_data)} FETs successfully")
    print(f"{'='*50}\n")
    
    return processed_data


def save_to_csv(data_dict, output_dir, prefix=""):
    """
    Save dictionary of dataframes to individual CSV files.
    
    Args:
        data_dict: Dictionary with FET IDs as keys and dataframes as values
        output_dir: Directory to save CSV files
        prefix: Optional prefix for filenames (e.g., "merged", "processed")
        
    Returns:
        None
    """
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    print("\n" + "="*50)
    print(f"Saving data to CSV files")
    print("="*50 + "\n")
    print(f"Output directory: {output_path}\n")
    
    saved_count = 0
    
    for fet_id, df in data_dict.items():
        # Remove metadata columns before saving
        cols_to_remove = ['source_file', 'sweep_type']
        df_to_save = df.drop(columns=[col for col in cols_to_remove if col in df.columns])
        
        # Create filename
        if prefix:
            filename = f"{fet_id}_{prefix}.csv"
        else:
            filename = f"{fet_id}.csv"
        
        output_file = output_path / filename
        
        # Save to CSV
        df_to_save.to_csv(output_file, index=False)
        
        print(f"Saved: {filename} ({len(df_to_save)} rows, {len(df_to_save.columns)} columns)")
        saved_count += 1
    
    print(f"\n{'='*50}")
    print(f"Successfully saved {saved_count} files to {output_path}")
    print(f"{'='*50}\n")


def plot_all_fets(processed_data, save_path=None, id_scale='log', ig_scale='linear', title='FET Transfer Characteristics - All Devices'):
    """
    Plot drain current and gate current vs Vg for all FETs on one graph.
    Drain current on left y-axis, gate current on right y-axis.
    Each FET uses the same color for both Id and Ig curves.
    
    Args:
        processed_data: Dictionary from process_fet_data()
        save_path: Optional path to save the figure (e.g., "./output/all_fets.png")
        id_scale: Scale for drain current y-axis ('log' or 'linear'), default='log'
        ig_scale: Scale for gate current y-axis ('log' or 'linear'), default='linear'
        title: Title of the plot, default='FET Transfer Characteristics - All Devices'
        
    Returns:
        None
    """
    # Create figure with dual y-axes
    fig, ax1 = plt.subplots(figsize=(12, 8))
    ax2 = ax1.twinx()
    
    # Color cycle for different FETs
    colors = plt.cm.tab10(np.linspace(0, 1, len(processed_data)))
    
    print("\n" + "="*50)
    print("Plotting all FETs")
    print("="*50 + "\n")
    
    for idx, (fet_id, df) in enumerate(sorted(processed_data.items())):
        color = colors[idx]
        
        # Check if required columns exist
        if 'Vg' not in df.columns:
            print(f"{fet_id}: Skipping - 'Vg' column not found")
            continue
        
        vg = df['Vg']
        
        # Plot drain current (left axis)
        if 'Drain Current (nA)' in df.columns:
            ax1.plot(vg, df['Drain Current (nA)'], 
                    color=color, linestyle='-', linewidth=2,
                    label=f'{fet_id} - Id')
        else:
            print(f"{fet_id}: Warning - 'Drain Current (nA)' column not found")
        
        # Plot gate current (right axis)
        if '|IG| (pA)' in df.columns:
            ax2.plot(vg, df['|IG| (pA)'], 
                    color=color, linestyle='--', linewidth=2,
                    label=f'{fet_id} - Ig')
        else:
            print(f"{fet_id}: Warning - '|IG| (pA)' column not found")
        
        print(f"Plotted: {fet_id}")
    
    # Configure left y-axis (drain current)
    ax1.set_xlabel('Gate Voltage Vg (V)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Drain Current (nA)', fontsize=12, fontweight='bold', color='black')
    ax1.set_yscale(id_scale)  # Set scale based on parameter
    ax1.set_ylim(1e-4, 10000)  # Fix scale to 0-10000 nA
    ax1.tick_params(axis='y', labelcolor='black', which='both')
    ax1.minorticks_on()
    ax1.tick_params(axis='y', which='minor', length=4)
    ax1.tick_params(axis='y', which='major', length=8)
    ax1.grid(True, alpha=0.3, which='major')
    ax1.grid(True, alpha=0.15, which='minor')
    
    # Configure right y-axis (gate current)
    ax2.set_ylabel('Gate Current (pA)', fontsize=12, fontweight='bold', color='black')
    ax2.set_yscale(ig_scale)  # Set scale based on parameter
    ax2.set_ylim(0, 1000)  # Fix scale to 0-1000 pA
    ax2.tick_params(axis='y', labelcolor='black', which='both')
    ax2.minorticks_on()
    ax2.tick_params(axis='y', which='minor', length=4)
    ax2.tick_params(axis='y', which='major', length=8)
    
    # Add title
    plt.title(title, fontsize=14, fontweight='bold', pad=20)
    
    # Combine legends from both axes
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, 
              loc='best', fontsize=9, framealpha=0.9)
    
    # Tight layout
    plt.tight_layout()
    
    # Save if path provided
    if save_path:
        save_file = Path(save_path)
        save_file.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_file, dpi=300, bbox_inches='tight')
        print(f"\nFigure saved to: {save_file}")
    
    plt.show()
    
    print(f"\n{'='*50}")
    print(f"Plotting complete")
    print(f"{'='*50}\n")