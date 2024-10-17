import argparse
import pandas as pd


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process input arguments.')
    parser.add_argument('--input_file', type=str,
                        default='/projects/datatrans/CarolinaOpenHealthData/full_set_results/output/unc_omop_2018_2022_kg.csv',
                        help='input csv file')
    parser.add_argument('--sample_size', default=100, help='sample size')
    parser.add_argument('--output_file', type=str,
                        default='/projects/datatrans/CarolinaOpenHealthData/full_set_results/output/unc_omop_2018_2022_kg_sampled.csv',
                        help='output csv file')

    args = parser.parse_args()
    input_file = args.input_file
    sample_size = args.sample_size
    output_file = args.output_file

    input_df = pd.read_csv(input_file)
    sampled_df = input_df.sample(n=sample_size, random_state=42)
    sampled_df.to_csv(output_file, index=False)
    print(f'sampled data file {output_file} is created')
    exit(0)
