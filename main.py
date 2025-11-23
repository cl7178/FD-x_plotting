
"""
Main script for FET data processing
Import and use functions from fet_processor module
"""

from IdVg_data_processor import (
    load_and_group_data,
    merge_sweeps,
    process_fet_data,
    save_to_csv,
    plot_all_fets
)
# Example usage
if __name__ == "__main__":
    # Set your data directory path
    data_directory = "./test_input_data"  # Change this to your folder path
    
    # Load and group data
    fet_data = load_and_group_data(data_directory)
    output_directory = "./merged_data"  # Change this to your desired output folder
    merged_data = merge_sweeps(fet_data)

    
    # Process the merged data
    processed_data = process_fet_data(merged_data)
    
    # Save data to CSV - uncomment the ones you want to save
    # save_to_csv(merged_data, "./merged_data", prefix="merged")
    save_to_csv(processed_data, "./processed_data", prefix="processed")
    
    # Plot all FETs
    #plot_all_fets(processed_data)

    # Or save the plot
    # Linear Id, log Ig
    plot_all_fets(processed_data, 
              save_path="./plots/test.png",
              id_scale='log', 
              ig_scale='linear',
              title='Test plot')
    # if 'FD-1' in fet_data:
    #     print(f"FD-1 has {len(fet_data['FD-1'])} dataframe(s)")
    #     print("\nFirst file data:")
    #     print(fet_data['FD-1'][0].head())