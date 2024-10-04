import pandas as pd
import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process input arguments.')
    parser.add_argument('--input_concept_pair_count_file', type=str,
                        default='data/concept_pair_counts_2018-2022_randomized_mincount-11_N-2306126_hierarchical_20240826-1228.txt',
                        help='input tsv file that includes concept pair counts')
    parser.add_argument('--unmapped_id_file', type=str,
                        default='mapping/unmapped_ids.txt',
                        help='input csv that includes omop concept ids that cannot be mapped to biolink ids')
    parser.add_argument('--output_file', type=str,
                        default='data/concept_pair_counts_2018-2022_randomized_mincount-11_N-2306126_hierarchical_20240826-1228_mapped.txt',
                        help='output concept pair counts file with unmapped rows filtered out')

    args = parser.parse_args()
    input_concept_pair_count_file = args.input_concept_pair_count_file
    unmapped_id_file = args.unmapped_id_file
    output_file = args.output_file

    unmapped_concept_df = pd.read_csv(unmapped_id_file, usecols=['unmapped_concept_id'], dtype=str)
    data_df = pd.read_csv(input_concept_pair_count_file, sep='\t', usecols=['concept_id1', 'concept_id2', 'count'],
                          dtype={'concept_id1': str, 'concept_id2': str, 'count': int})
    unmapped_concept_ids = set(unmapped_concept_df['unmapped_concept_id'])

    filtered_data_df = data_df[
        (~data_df['concept_id1'].isin(unmapped_concept_ids)) &
        (~data_df['concept_id2'].isin(unmapped_concept_ids))
    ]

    filtered_data_df.to_csv(output_file, index=False, sep='\t')
