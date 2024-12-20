########################################################
# This script is adapted from KGX COHD implementation
# https://github.com/WengLab-InformaticsResearch/cohd_api/blob/master/kgx/kgx_cohd.py
########################################################
import os
import gc
import argparse
import numpy as np
from scipy.stats import poisson, chisquare
from datetime import datetime
import pandas as pd
from utils import normalize_nodes

THRESHOLD_COUNT = 10
LN_RATIO_THRESHOLD = 1.0
MIN_P = 1e-12  # ARAX displays p-value of 0 as None. Replace with a minimum p-value
omop_ids_not_mapped = []

# Pre-cache values for poisson_ci. Confidence values of 0.99 and 0.999 are commonly used. Caching up to a freq of 10000
# covers 99% of co-occurrence counts in COHD and takes up < 1MB RAM for both confidence levels.
_poisson_ci_cache = {
    0.99: dict(),
    0.999: dict()
}

def poisson_ci(freq, confidence=0.99):
    """ Assuming two Poisson processes (1 for the event rate and 1 for randomization), calculate the confidence interval
    for the true rate

    Parameters
    ----------
    freq: float - co-occurrence frequency
    confidence: float - desired confidence. range: [0, 1]

    Returns
    -------
    (lower bound, upper bound)
    """
    # # Adjust the interval for each individual poisson to achieve overall confidence interval
    # return poisson.interval(confidence, freq)

    # COHD defaults to confidence values of 0.99 and 0.999 (double poisson), so cache these values to save compute time
    use_cache = (confidence == 0.99 or confidence == 0.999)
    if use_cache:
        cache = _poisson_ci_cache[confidence]
        if freq in cache:
            return cache[freq]

    # Same result as using poisson.interval, but much faster calculation
    alpha = 1 - confidence
    ci = poisson.ppf([alpha / 2, 1 - alpha / 2], freq)
    ci[0] = max(ci[0], 1)  # min possible count is 1
    ci = tuple(ci)

    if use_cache:
        # Only cache results for 99% and 99.9% CI
        _poisson_ci_cache[confidence][freq] = ci
    return ci


def double_poisson_ci(freq, confidence=0.99):
    """ Assuming two Poisson processes (1 for the event rate and 1 for randomization), calculate the confidence interval
    for the true rate

    Parameters
    ----------
    freq: float - co-occurrence frequency
    confidence: float - desired confidence. range: [0, 1]

    Returns
    -------
    (lower bound, upper bound)
    """
    # # Adjust the interval for each individual poisson to achieve overall confidence interval
    # confidence_adjusted = 1 - (1 - confidence) ** 0.5
    # return (poisson.interval(confidence_adjusted, poisson.interval(confidence_adjusted, freq)[0])[0],
    #         poisson.interval(confidence_adjusted, poisson.interval(confidence_adjusted, freq)[1])[1])

    # More efficient calculation using a single call to poisson.interval with similar results as above
    # Adjust the interval for each individual poisson to achieve overall confidence interval
    confidence_adjusted = 1 - ((1 - confidence) ** 1.5)
    return poisson_ci(freq, confidence_adjusted)


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


def compute_log_odds(c1, c2, cp, n):
    """ Calculates log-odds and 95% CI

    Params
    ------
    c1: count for concept 1
    c2: count for concept 2
    cp: concept-pair count
    n: total population size

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
            return np.inf, [np.inf, np.inf]
    else:
        log_odds_val = np.log((a*d)/(b*c))
        ci = 1.96 * np.sqrt(1/a + 1/b + 1/c + 1/d)
        ci = [log_odds_val - ci, log_odds_val + ci]
        return log_odds_val, ci


def chi_square(cpc, c1, c2, pts, min_p=MIN_P):
    """ Calculate p-value using Chi-square

    Params
    ------
    cpc: concept-pair count
    c1: count for concept 1
    c2: count for concept 2
    pts: total population size
    min_p: minimum p-value to return
    """
    neg = pts - c1 - c2 + cpc
    # Create the observed and expected RxC tables and perform chi-square
    o = [neg, c1 - cpc, c2 - cpc, cpc]
    e = [(pts - c1) * (pts - c2) / pts, c1 * (pts - c2) / pts, c2 * (pts - c1) / pts, c1 * c2 / pts]
    cs = chisquare(o, e, 2)
    p = max(cs.pvalue, min_p)
    return p


def ln_ratio_ci(freq, ln_ratio, confidence=0.99):
    """ Estimates the confidence interval of the log ratio using the double poisson method

    Parameters
    ----------
    freq: float - co-occurrence count
    ln_ratio: float - log ratio
    confidence: float - desired confidence. range: [0, 1]

    Returns
    -------
    (lower bound, upper bound)
    """
    # Convert ln_ratio back to ratio and calculate confidence intervals for the ratios
    ci = tuple(np.log(np.array(double_poisson_ci(freq, confidence)) * np.exp(ln_ratio) / freq))
    return ci


def compute_edge_info(row, map_df, total_pats):
    # return subject, subject_name, object, object_name, predicate, chi_squared_p_value,
    # log_odds_ratio, log_odds_ratio_95_ci, count_pair
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

    if count_1 < THRESHOLD_COUNT or count_2 < THRESHOLD_COUNT or count_pair < THRESHOLD_COUNT:
        # don't return edge info if counts are less than THRESHOLD_COUNT to protect patient privacy
        return biolink_id_1, biolink_label_1, biolink_id_2, biolink_label_2, None, None, None, None, None

    # calculate ln_ratio
    lnr = np.log(count_pair * total_pats / (count_1 * count_2))
    lnr_ci = ln_ratio_ci(count_pair, lnr)

    if lnr_ci[0] > LN_RATIO_THRESHOLD or lnr_ci[1] < -LN_RATIO_THRESHOLD:
        # calculate chi-square
        p_val = chi_square(count_pair, count_1, count_2, total_pats)
        # calculate log-odds
        lo, lo_ci = compute_log_odds(count_1, count_2, count_pair, total_pats)
        score = lnr_ci[0] if lnr > 0 else -lnr_ci[1]
        predicate = 'biolink:positively_correlated_with' if lo_ci[0] > 0 else 'biolink:negatively_correlated_with'
        return biolink_id_1, biolink_label_1, biolink_id_2, biolink_label_2, predicate, p_val, lo, lo_ci, score
    else:
        return biolink_id_1, biolink_label_1, biolink_id_2, biolink_label_2, None, None, None, None, None

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
                   'log_odds_ratio_95_ci', 'score']] \
            = chunk_df.apply(lambda row: compute_edge_info(row, mapping_df, patient_num), axis=1, result_type='expand')
        chunk_df.drop(columns=['concept_id1', 'concept_id2', 'count_pair', 'count_concept_id1', 'count_concept_id2'],
                      inplace=True)
        # filter out those rows with predicate column N/A
        chunk_df = chunk_df[chunk_df['predicate'].notna()]
        chunk_df.to_csv(output_file, index=False, float_format="%.3f")

    if omop_ids_not_mapped:
        omop_ids_not_mapped = list(set(omop_ids_not_mapped))
        with open(f'{os.path.splitext(input_mapping_file)[0]}_omop_ids_not_mapped.txt', 'w') as f:
            f.writelines([f'{line}\n' for line in omop_ids_not_mapped])

    print(f'knowledge graph chunks are created, time taken: {(datetime.now() - t1).seconds}')
