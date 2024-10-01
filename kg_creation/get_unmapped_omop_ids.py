import pandas as pd
import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process input arguments.')
    parser.add_argument('--input_concept_pair_count_file', type=str,
                        default='data/concept_pair_counts_2018-2022_randomized_mincount-11_N-2306126_hierarchical_20240826-1228.txt',
                        help='input tsv file that includes concept pair counts')
    parser.add_argument('--input_mapping_file', type=str,
                        default='mapping/mappings.csv',
                        help='input csv that includes mappings from OMOP concept id to biolink id')
    parser.add_argument('--output_file', type=str,
                        default='mapping/unmapped_id.txt',
                        help='output file containing unmapped OMOP concept ids')

    args = parser.parse_args()
    input_concept_pair_count_file = args.input_concept_pair_count_file
    input_mapping_file = args.input_mapping_file
    output_file = args.output_file

    mapping_df = pd.read_csv(input_mapping_file, usecols=['omop_id'], dtype=str)
    data_df = pd.read_csv(input_concept_pair_count_file, sep='\t', usecols=['concept_id1', 'concept_id2'], dtype=str)
    unique_concept_ids = pd.concat([data_df['concept_id1'], data_df['concept_id2']]).unique()
    omop_ids_set = set(mapping_df['omop_id'])
    unmapped_concept_ids = [concept_id for concept_id in unique_concept_ids if concept_id not in omop_ids_set]
    unmapped_concept_df = pd.DataFrame(unmapped_concept_ids, columns=['filtered_concept_id'])
    unmapped_concept_df.to_csv(output_file, index=False)
