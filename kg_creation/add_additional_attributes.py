import argparse
import pandas as pd


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process input arguments.')
    parser.add_argument('--input_file', type=str,
                        default='/projects/datatrans/CarolinaOpenHealthData/full_set_results/output/unc_omop_2018_2022_kg.csv',
                        help='input csv file')
    parser.add_argument('--added_attrs',
                        default={
                            'total_sample_size': 2344578,
                            'primary_knowledge_source': 'infores:icees-kg',
                            'aggregator_knowledge_source': 'infores:automat-openhealthdata-carolina'
                        },
                        help='static attributes that need to be added')
    parser.add_argument('--output_file', type=str,
                        default='/projects/datatrans/CarolinaOpenHealthData/full_set_results/output/unc_omop_2018_2022_kg_updated.csv',
                        help='output csv file')

    args = parser.parse_args()
    input_file = args.input_file
    added_attrs = args.added_attrs
    output_file = args.output_file

    input_df = pd.read_csv(input_file)
    print(f'input csv dataframe shape: {input_df.shape}')
    if 'log_odds_ratio_95_confidence_interval' in input_df.columns:
        input_df.rename(columns={'log_odds_ratio_95_confidence_interval': 'log_odds_ratio_95_ci'}, inplace=True)
    for k, v in added_attrs.items():
        input_df[k] = v
    input_df.to_csv(output_file, index=False)
    print(f'updated csv dataframe shape: {input_df.shape}')
    exit(0)
