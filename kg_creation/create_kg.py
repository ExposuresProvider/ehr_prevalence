import os
import argparse
import numpy as np
from datetime import datetime
import pandas as pd
from utils import normalize_nodes


def get_normalized_id_and_name(row_id, map_dict):
    mapped_data = map_dict.get(row_id, {})
    if mapped_data:
        equiv_ids = [eq_id['identifier'] for eq_id in mapped_data['equivalent_identifiers']]
        equiv_labels = [eq_id['label'] for eq_id in mapped_data['equivalent_identifiers']]
        id_to_label = {}
        for eid, elabel in zip(equiv_ids, equiv_labels):
            id_to_label[eid] = elabel
        if len(equiv_ids) <= 0 or (row_id in equiv_ids):
            return row_id, id_to_label[row_id]
        # if not found, return the first equivalent identifier in the list
        return equiv_ids[0], id_to_label[equiv_ids[0]]
    return row_id, ''


def log_odds(c1, c2, cp, n, replace_inf=np.inf):
    """ Calculates the log-odds and 95% CI 

    Params
    ------
    c1: count for concept 1
    c2: count for concept 2
    cp: concept-pair count
    n: total population size
    replace_inf: (Optional) If specified, replaces +Inf or -Inf with +replace_inf or -replace_inf (useful because JSON
                 doesn't allow Infinity)

    Returns
    -------
    (log-odds, [95% CI lower bound, 95% CI upper bound])
    """
    a = cp
    b = c1 - cp
    c = c2 - cp
    d = n - c1 - c2 + cp
    # Check b/c <= 0 since Poisson perturbation can cause b or c to be negative
    if b <= 0 or c <= 0:
        if a == 0:
            return 0, [0, 0]
        else:
            return replace_inf, [replace_inf, replace_inf]
    else:
        log_odds_val = np.log((a*d)/(b*c))
        ci = 1.96 * np.sqrt(1/a + 1/b + 1/c + 1/d)
        ci = [clip(log_odds_val - ci, replace_inf), clip(log_odds_val + ci, replace_inf)]
        return clip(log_odds_val, replace_inf), ci


def clip(x, clip_val):
    """ Clip values to [-clip_val, clip_val]
    
    Params
    ------
    x: value to clip_val
    clip_val: value to clip to
    
    Returns
    -------
    clipped value 
    """
    # return min(max(x, -clip), clip)
    return -clip_val if x < -clip_val else clip_val if x > clip_val else x


def compute_edge_info(row, map_df, total_pats):
    # return subject, object, predicate, log_odds_ratio, log_odds_ratio_95_ci
    omop_id_1, omop_id_2, count_1, count_2, count_pair = (row['concept_id1'], row['concept_id2'],
                                                          row['count_concept_id1'], row['count_concept_id2'],
                                                          row['count_pair'])
    map_row_1 = map_df[map_df['omop_id'] == omop_id_1]
    map_row_2 = map_df[map_df['omop_id'] == omop_id_2]
    if not map_row_1.empty:
        biolink_id_1 = map_row_1.iloc[0]['normalized_biolink_id']
        biolink_label_1 = map_row_1.iloc[0]['normalized_biolink_label']
    else:
        biolink_id_1 = biolink_label_1 = None
    if not map_row_2.empty:
        biolink_id_2 = map_row_2.iloc[0]['normalized_biolink_id']
        biolink_label_2 = map_row_2.iloc[0]['normalized_biolink_label']
    else:
        biolink_id_2 = biolink_label_2 = None

    if count_1 < 10 or count_2 < 10 or count_pair < 10:
        return biolink_id_1, biolink_label_1, biolink_id_2, biolink_label_2, None, None, None, None, count_pair

    # calculate log-odds
    lo, lo_ci = log_odds(count_1, count_2, count_pair, total_pats)

    if lo_ci[0] > 0.5 or lo_ci[1] < -0.5:
        predicate = 'biolink:positively_correlated_with' if lo_ci[0] > 0 else 'biolink:negatively_correlated_with'
        return biolink_id_1, biolink_label_1, biolink_id_2, biolink_label_2, predicate, lo, lo_ci[0], lo_ci[1], count_pair
    else:
        return biolink_id_1, biolink_label_1, biolink_id_2, biolink_label_2, None, None, None, None, count_pair


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process input arguments.')
    parser.add_argument('--input_concept_count_file', type=str,
                        default='kg_creation/data/concept_counts_2018-2022_randomized_mincount-11_N-2306126_hierarchical_20240826-1228.txt',
                        help='input tsv file that includes concept counts')
    parser.add_argument('--input_concept_pair_count_file', type=str,
                        default='kg_creation/data/concept_pair_counts_2018-2022_randomized_mincount-11_N-2306126_hierarchical_20240826-1228.txt',
                        help='input tsv file that includes concept pair counts')
    parser.add_argument('--input_mapping_file', type=str,
                        default='kg_creation/mapping/mappings.csv',
                        help='input csv that includes mappings from OMOP concept id to biolink id')
    parser.add_argument('--patient_num', type=int, default=1731858, help='number of total patients in the data')
    parser.add_argument('--output_kg_file', type=str,
                        default='kg_creation/data/unc_omop_2018_2022_kg.csv',
                        help='output csv that includes knowledge graph csv created from concept count and concept pair counts')

    args = parser.parse_args()
    input_concept_count_file = args.input_concept_count_file
    input_concept_pair_count_file = args.input_concept_pair_count_file
    patient_num = args.patient_num
    input_mapping_file = args.input_mapping_file
    output_kg_file = args.output_kg_file

    t1 = datetime.now()
    mapping_df = pd.read_csv(input_mapping_file)
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

    input_concept_count_df = pd.read_csv(input_concept_count_file, sep='\t', usecols=['concept_id', 'count'])
    input_concept_pair_count_df = pd.read_csv(input_concept_pair_count_file, sep='\t',
                                              usecols=['concept_id1', 'concept_id2', 'count'])

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

    t1 = datetime.now()
    joined_df[['subject', 'subject_name', 'object', 'object_name', 'predicate', 'log_odds_ratio',
               'log_odds_ratio_95_ci_lower', 'log_odds_ratio_95_ci_lower', 'total_sample_size']] \
        = joined_df.apply(lambda row: compute_edge_info(row, mapping_df, patient_num), axis=1, result_type='expand')
    print(f'knowledge graph is created, time taken: {(datetime.now() - t1).seconds}')
