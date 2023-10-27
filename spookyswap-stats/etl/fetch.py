#   -*- coding: utf-8 -*-
#   ------------------------------------------------------------------------------
#
#     Copyright 2021 Valory AG
#
#     Licensed under the Apache License, Version 2.0 (the "License");
#     you may not use this file except in compliance with the License.
#     You may obtain a copy of the License at
#
#         http://www.apache.org/licenses/LICENSE-2.0
#
#     Unless required by applicable law or agreed to in writing, software
#     distributed under the License is distributed on an "AS IS" BASIS,
#     WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#     See the License for the specific language governing permissions and
#     limitations under the License.
#
#   ------------------------------------------------------------------------------

"""Fetching from subgraph operations."""
import functools
import warnings
from typing import Callable, List

import requests
from tqdm import tqdm

from config.general import SPOOKY_SUBGRAPH_URL, FANTOM_BLOCKS_SUBGRAPH_URL, BUNDLE_ID
from etl import ResponseItemType, SubgraphResponseType
from etl.queries import block_from_timestamp_q, pairs_q, eth_price_usd_q
from etl.tools import gen_unix_timestamps, sec_to_unit


class RetriesExceeded(Exception):
    """Exception to raise when retries are exceeded during data-fetching."""
    def __init__(self, msg='Maximum retries were exceeded while trying to fetch the data!'):
        super().__init__(msg)


def hacky_retry(func: Callable, n_retries: int = 3) -> Callable:
    """Create a hacky retry strategy.
        Unfortunately, we cannot use `requests.packages.urllib3.util.retry.Retry`,
        because the subgraph does not return the appropriate status codes in case of failure.
        Instead, it always returns code 200. Thus, we raise exceptions manually inside `make_request`,
        catch those exceptions in the hacky retry decorator and try again.
        Finally, if the allowed number of retries is exceeded, we raise a custom `RetriesExceeded` exception.

    :param func: the input request function.
    :param n_retries: the maximum allowed number of retries.
    :return: The request method with the hacky retry strategy applied.
    """
    @functools.wraps(func)
    def wrapper_hacky_retry(*args, **kwargs) -> SubgraphResponseType:
        """The wrapper for the hacky retry.

        :return: a response dictionary.
        """
        retried = 0

        while retried <= n_retries:
            try:
                if retried > 0:
                    warnings.warn(f"Retrying {retried}/{n_retries}...")

                return func(*args, **kwargs)
            except (ValueError, ConnectionError) as e:
                warnings.warn(e.args[0])
            finally:
                retried += 1

        raise RetriesExceeded()

    return wrapper_hacky_retry


@hacky_retry
def make_request(url: str, query_fn: Callable[..., str], *query_args) -> SubgraphResponseType:
    """Make a request to a subgraph.

    Args:
        url: the subgraph's URL.
        query_fn: the query function to be used.
        *query_args: arguments for the query function.

    Returns:
        a response dictionary.
    """
    r = requests.post(url, json={'query': query_fn(*query_args)})

    if r.status_code == 200:
        res = r.json()

        if 'errors' in res.keys():
            message = res['errors'][0]['message']
            location = res['errors'][0]['locations'][0]
            line = location['line']
            column = location['column']

            raise ValueError(f'The given query is not correct.\nError in line {line}, column {column}: {message}')

        elif 'data' not in res.keys():
            raise ValueError(f'Unknown error encountered!\nRaw response: \n{res}')

    else:
        raise ConnectionError('Something went wrong while trying to communicate with the subgraph '
                              f'(Error: {r.status_code})!\n{r.text}')

    return res['data']


def get_pairs_hist(pair_ids: List[str], start: int, interval_in_unix: int, end: int) -> ResponseItemType:
    """Get historical data for the given pools.

    :param pair_ids: the ids of the pairs to fetch.
    :param start: the start date in Unix timestamp.
    :param interval_in_unix: the interval in Unix to use for the fetched data.
    :param end: the end date in Unix timestamp.
    :return: a list with the historical data of the pairs, retrieved from SpookySwap's subgraph.
    """
    pairs_hist = []
    timestamps_generator = gen_unix_timestamps(start, interval_in_unix, end)
    n_iter = int((end - start) / interval_in_unix)
    interval_unit = sec_to_unit(interval_in_unix)

    for timestamp in tqdm(timestamps_generator, total=n_iter, desc='Fetching historical data', unit=interval_unit):
        # Fetch block.
        res = make_request(FANTOM_BLOCKS_SUBGRAPH_URL, block_from_timestamp_q, timestamp)
        fetched_block = res['blocks'][0]

        # Fetch ETH price for block.
        res = make_request(SPOOKY_SUBGRAPH_URL, eth_price_usd_q, *(BUNDLE_ID, fetched_block['number']))
        eth_price = float(res['bundles'][0]['ethPrice'])

        # Fetch top n pool data for block.
        res = make_request(SPOOKY_SUBGRAPH_URL, pairs_q, *(fetched_block['number'], pair_ids))

        # Add extra fields to the pairs.
        for i in range(len(res['pairs'])):
            res['pairs'][i]['for_timestamp'] = timestamp
            res['pairs'][i]['block_number'] = fetched_block['number']
            res['pairs'][i]['block_timestamp'] = fetched_block['timestamp']
            res['pairs'][i]['eth_price'] = str(eth_price)

        pairs_hist.extend(res['pairs'])

    return pairs_hist
