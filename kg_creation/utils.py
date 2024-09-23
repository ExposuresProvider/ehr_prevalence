import requests

NODENORM_URL = 'https://nodenormalization-sri.renci.org/1.5/get_normalized_nodes'


def normalize_nodes(input_list, request_list_size=100):
    """
    Use nodenorm service to normalize identifiers in input_list
    :param input_list: non-duplicate set of identifiers to be normalized
    :param request_list_size: optional parameter with default value 100 which specifies the number of
    identifiers to send at one time for the POST request to get normalized nodes for.
    :return: dict with identifier to be normalized as key and equivalent identifiers as value
    """
    result_dict = {}
    count = 0
    set_size = len(input_list)
    while count < set_size:
        low_bound = count
        high_bound = low_bound + request_list_size
        try:
            result = requests.post(NODENORM_URL, json={
                "curies": input_list[low_bound:high_bound]
            })
            result_dict.update(result.json())
        except Exception as ex:
            print(f'count: {count}, input_list: {input_list[low_bound:high_bound]}, exception: {ex}')
        count += request_list_size
    return result_dict

