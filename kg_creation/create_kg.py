import os
import gc
import argparse
import numpy as np
from scipy.stats import chi2_contingency
from datetime import datetime
import pandas as pd
from utils import normalize_nodes

omop_ids_not_mapped = []

def get_normalized_id_and_name(row_id, map_dict):
    mapped_data = map_dict.get(row_id, {})
    eid = label = ''
    if mapped_data:
        for eq_id in mapped_data['equivalent_identifiers']:
            if not eid:
                eid = eq_id.get('identifier', '')
            if not label:
                label = eq_id.get('label', '')
            if eid and label:
                break

    if eid and label:
        return eid, label

    return row_id, ''


def compute_stats(c1, c2, cp, n):
    """ Calculates the p-value, log-odds and 95% CI

    Params
    ------
    c1: count for concept 1
    c2: count for concept 2
    cp: concept-pair count
    n: total population size

    Returns
    -------
    (p_value, log-odds, [95% CI lower bound, 95% CI upper bound])
    """
    a = cp
    b = c1 - cp
    c = c2 - cp
    d = n - c1 - c2 + cp

    # Check b/c <= 0 since Poisson perturbation can cause b or c to be negative
    if b <= 0 or c <= 0:
        if a == 0:
            return 0, 0, [0, 0]
        else:
            return np.inf, np.inf, [np.inf, np.inf]
    else:
        try:
            contingency_table = np.array([[a, b], [c, d]])
            _, p_value, _, _ = chi2_contingency(contingency_table, correction=False)
        except ValueError as ex:
            print(f'a: {a}, b: {b}, c: {c}, d: {d}, exception: {ex}')
            p_value = np.inf
        log_odds_val = np.log((a*d)/(b*c))
        ci = 1.96 * np.sqrt(1/a + 1/b + 1/c + 1/d)
        ci = [log_odds_val - ci, log_odds_val + ci]
        return p_value, log_odds_val, ci


def compute_edge_info(row, map_df, total_pats):
    # return subject, subject_name, object, object_name, predicate, chi_squared_p_value,
    # log_odds_ratio, log_odds_ratio_95_confidence_interval, count_pair
    omop_id_1, omop_id_2, count_1, count_2, count_pair = (row['concept_id1'], row['concept_id2'],
                                                          row['count_concept_id1'], row['count_concept_id2'],
                                                          row['count_pair'])
    map_row_1 = map_df[map_df['omop_id'] == omop_id_1]
    map_row_2 = map_df[map_df['omop_id'] == omop_id_2]
    if not map_row_1.empty:
        biolink_id_1 = map_row_1.iloc[0]['normalized_biolink_id'] if map_row_1.iloc[0]['normalized_biolink_id'] \
            else map_row_1.iloc[0]['biolink_id']
        biolink_label_1 = map_row_1.iloc[0]['normalized_biolink_label']
    else:
        omop_ids_not_mapped.append(omop_id_1)
        biolink_id_1 = biolink_label_1 = ''
    if not map_row_2.empty:
        biolink_id_2 = map_row_2.iloc[0]['normalized_biolink_id'] if map_row_2.iloc[0]['normalized_biolink_id'] \
            else map_row_2.iloc[0]['biolink_id']
        biolink_label_2 = map_row_2.iloc[0]['normalized_biolink_label']
    else:
        biolink_id_2 = biolink_label_2 = ''
        omop_ids_not_mapped.append(omop_id_2)

    if count_1 < 10 or count_2 < 10 or count_pair < 10:
        # don't return edge info if counts are less than 10 to protect patient privacy
        return biolink_id_1, biolink_label_1, biolink_id_2, biolink_label_2, None, None, None, None

    # calculate log-odds
    p_val, lo, lo_ci = compute_stats(count_1, count_2, count_pair, total_pats)
    if lo_ci[0] > 0.5 or lo_ci[1] < -0.5:
        predicate = 'biolink:positively_correlated_with' if lo_ci[0] > 0 else 'biolink:negatively_correlated_with'
        return biolink_id_1, biolink_label_1, biolink_id_2, biolink_label_2, predicate, p_val, lo, lo_ci
    else:
        return biolink_id_1, biolink_label_1, biolink_id_2, biolink_label_2, None, None, None, None


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process input arguments.')
    parser.add_argument('--input_concept_count_file', type=str,
                        default='/projects/datatrans/CarolinaOpenHealthData/full_set_results/'
                                'concept_counts_2018-2022_randomized_mincount-11_N-2306126_hierarchical_20240826-1228.txt',
                        help='input tsv file that includes concept counts')
    parser.add_argument('--input_concept_pair_count_file', type=str,
                        default='/projects/datatrans/CarolinaOpenHealthData/full_set_results/'
                                'concept_pair_counts_2018-2022_randomized_mincount-11_N-2306126_hierarchical_20240826-1228.txt',
                        help='input tsv file that includes concept pair counts')
    parser.add_argument('--input_mapping_file', type=str,
                        # default='mapping/mappings.csv',
                        default='/projects/datatrans/CarolinaOpenHealthData/mapping/mappings.csv',
                        help='input csv that includes mappings from OMOP concept id to biolink id')
    parser.add_argument('--patient_num', type=int, default=2344578, help='number of total patients in the data')
    parser.add_argument('--output_kg_chunk_base', type=str,
                        default='/projects/datatrans/CarolinaOpenHealthData/full_set_results/output/chunks/unc_omop_2018_2022_kg',
                        help='output csv that includes knowledge graph csv created from concept count and concept pair counts')

    args = parser.parse_args()
    input_concept_count_file = args.input_concept_count_file
    input_concept_pair_count_file = args.input_concept_pair_count_file
    patient_num = args.patient_num
    input_mapping_file = args.input_mapping_file
    output_kg_chunk_base = args.output_kg_chunk_base

    t1 = datetime.now()
    mapping_df = pd.read_csv(input_mapping_file, dtype=str)
    if 'omop_id' not in mapping_df.columns or 'biolink_id' not in mapping_df.columns:
        print(f'two required columns: omop_id and biolink_id, are not in the mapping input file {input_mapping_file}')
        exit(1)

    if 'normalized_biolink_id' not in mapping_df.columns:
        # call DT nodenorm service to normalize biolink ids
        unique_biolink_ids = list(set(mapping_df['biolink_id'].tolist()))
        print(f'unique biolink ids len: {len(unique_biolink_ids)}')
        id_mapping_dict = normalize_nodes(unique_biolink_ids)
        print(f'norm biolink ids result len: {len(id_mapping_dict)}')
        mapping_df[['normalized_biolink_id', 'normalized_biolink_label']] = mapping_df.apply(lambda row:
            get_normalized_id_and_name(row['biolink_id'], id_mapping_dict), axis=1, result_type='expand')
        mapping_df.to_csv(f'{os.path.splitext(input_mapping_file)[0]}_normalized.csv', index=False)
    print(f'mapping is loaded and normalized, time taken: {(datetime.now() - t1).seconds}')

    input_concept_count_df = pd.read_csv(input_concept_count_file, sep='\t', usecols=['concept_id', 'count'],
                                         dtype={'concept_id': str, 'count': int})
    input_concept_pair_count_df = pd.read_csv(input_concept_pair_count_file, sep='\t',
                                              usecols=['concept_id1', 'concept_id2', 'count'],
                                              dtype={'concept_id1': str, 'concept_id2': str, 'count': int})

    # Perform left join on 'concept_id1'
    joined_df = pd.merge(input_concept_pair_count_df, input_concept_count_df, left_on='concept_id1',
                         right_on='concept_id', how='left')
    # Rename the 'count' column from the right_df to reflect the join with 'concept_id1'
    joined_df.rename(columns={'count_y': 'count_concept_id1', 'count_x': 'count_pair'}, inplace=True)
    # Drop the unnecessary 'concept_id' column from the right_df
    joined_df.drop(columns=['concept_id'], inplace=True)

    # Perform left join on 'concept_id2'
    joined_df = pd.merge(joined_df, input_concept_count_df, left_on='concept_id2', right_on='concept_id', how='left')
    # Rename the 'count' column from the right_df to reflect the join with 'concept_id2'
    joined_df.rename(columns={'count': 'count_concept_id2'}, inplace=True)
    # Drop the unnecessary 'concept_id' column from the right_df
    joined_df.drop(columns=['concept_id'], inplace=True)
    print(f'joined_df shape: {joined_df.shape}', flush=True)
    del input_concept_count_df
    del input_concept_pair_count_df
    gc.collect()

    chunk_size = 1000000
    num_chunks = (len(joined_df) // chunk_size) + 1
    t1 = datetime.now()
    for chunk_idx in range(num_chunks):
        output_file = f'{output_kg_chunk_base}_chunk_{chunk_idx}.csv'
        start_idx = chunk_idx * chunk_size
        end_idx = min(start_idx + chunk_size, len(joined_df))
        chunk_df = joined_df.iloc[start_idx:end_idx].copy()
        print(f"Processing chunk {chunk_idx} ({start_idx} to {end_idx})...", flush=True)
        chunk_df[['subject', 'subject_name', 'object', 'object_name', 'predicate', 'chi_squared_p_value', 'log_odds_ratio',
                   'log_odds_ratio_95_confidence_interval']] \
            = chunk_df.apply(lambda row: compute_edge_info(row, mapping_df, patient_num), axis=1, result_type='expand')
        chunk_df.drop(columns=['concept_id1', 'concept_id2', 'count_pair', 'count_concept_id1', 'count_concept_id2'],
                      inplace=True)
        chunk_df.to_csv(output_file, index=False, float_format="%.3f")

    if omop_ids_not_mapped:
        omop_ids_not_mapped = list(set(omop_ids_not_mapped))
        with open(f'{os.path.splitext(input_mapping_file)[0]}_omop_ids_not_mapped.txt', 'w') as f:
            f.writelines([f'{line}\n' for line in omop_ids_not_mapped])

    print(f'knowledge graph chunks are created, time taken: {(datetime.now() - t1).seconds}')
