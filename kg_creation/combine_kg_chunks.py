import os
import argparse
import pandas as pd


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process input arguments.')
    parser.add_argument('--input_dir', type=str,
                        default='/projects/datatrans/CarolinaOpenHealthData/full_set_results/output/chunks',
                        help='input directory that contains chunk csv files to be combined')
    parser.add_argument('--output_file', type=str,
                        default='/projects/datatrans/CarolinaOpenHealthData/full_set_results/output/unc_omop_2018_2022_kg.csv',
                        help='combined output csv file')

    args = parser.parse_args()
    input_dir = args.input_dir
    output_file = args.output_file

    all_chunks = []
    for chunk_file in sorted(os.listdir(input_dir)):
        # Only consider CSV files (in case there are other files in the folder)
        if chunk_file.endswith(".csv"):
            chunk_path = os.path.join(input_dir, chunk_file)
            print(f"Reading {chunk_path}...", flush=True)
            # Read the chunk and append it to the list
            chunk_df = pd.read_csv(chunk_path, dtype=str)
            all_chunks.append(chunk_df)

    combined_df = pd.concat(all_chunks, ignore_index=True)
    combined_df.to_csv(output_file, index=False)
    print(f'combined knowledge graph file {output_file} is created')
