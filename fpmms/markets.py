#   -*- coding: utf-8 -*-
#   ------------------------------------------------------------------------------
#
#     Copyright 2023 Valory AG
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

import functools
import os.path
import warnings
from string import Template
from typing import Optional, Generator, Callable

import pandas as pd
import requests
from tqdm import tqdm

from typing import List, Dict


ResponseItemType = List[Dict[str, str]]
SubgraphResponseType = Dict[str, ResponseItemType]


CREATOR = "0x89c5cc945dd550BcFfb72Fe42BfF002429F46Fec"
BATCH_SIZE = 1000
OMEN_SUBGRAPH = "https://api.thegraph.com/subgraphs/name/protofire/omen-xdai"
FPMMS_FIELD = "fixedProductMarketMakers"
QUERY_FIELD = "query"
ERROR_FIELD = "errors"
DATA_FIELD = "data"
ID_FIELD = "id"
ANSWER_FIELD = "currentAnswer"
QUESTION_FIELD = "question"
OUTCOMES_FIELD = "outcomes"
TITLE_FIELD = "title"
MAX_UINT_HEX = "0xffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff"
DEFAULT_FILENAME = "fpmms.csv"

FPMMS_QUERY = Template(
    """
    {
      ${fpmms_field}(
        where: {
          creator: "${creator}",
          id_gt: "${fpmm_id}",
          isPendingArbitration: false
        },
        orderBy: ${id_field}
        first: ${first}
      ){
        ${id_field}
        ${answer_field}
        ${question_field} {
          ${outcomes_field}
        }
        ${title_field}
      }
    }
    """
)


class RetriesExceeded(Exception):
    """Exception to raise when retries are exceeded during data-fetching."""

    def __init__(
        self, msg="Maximum retries were exceeded while trying to fetch the data!"
    ):
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
def query_subgraph(url: str, query: str, key: str) -> SubgraphResponseType:
    """Query a subgraph.

    Args:
        url: the subgraph's URL.
        query: the query to be used.
        key: the key to use in order to access the required data.

    Returns:
        a response dictionary.
    """
    content = {QUERY_FIELD: query}
    headers = {
        "Accept": "application/json",
        "Content-Type": "application/json",
    }
    res = requests.post(url, json=content, headers=headers)

    if res.status_code != 200:
        raise ConnectionError(
            "Something went wrong while trying to communicate with the subgraph "
            f"(Error: {res.status_code})!\n{res.text}"
        )

    body = res.json()
    if ERROR_FIELD in body.keys():
        raise ValueError(f"The given query is not correct: {body[ERROR_FIELD]}")

    data = body.get(DATA_FIELD, {}).get(key, None)
    if data is None:
        raise ValueError(f"Unknown error encountered!\nRaw response: \n{body}")

    return data


def fpmms_fetcher() -> Generator[ResponseItemType, int, None]:
    """An indefinite fetcher for the FPMMs."""
    while True:
        fpmm_id = yield
        fpmms_query = FPMMS_QUERY.substitute(
            creator=CREATOR,
            fpmm_id=fpmm_id,
            fpmms_field=FPMMS_FIELD,
            first=BATCH_SIZE,
            id_field=ID_FIELD,
            answer_field=ANSWER_FIELD,
            question_field=QUESTION_FIELD,
            outcomes_field=OUTCOMES_FIELD,
            title_field=TITLE_FIELD,
        )
        yield query_subgraph(OMEN_SUBGRAPH, fpmms_query, FPMMS_FIELD)


def fetch_fpmms(latest_id: str) -> pd.DataFrame:
    """Fetch all the fpmms of the creator."""
    fpmms = []
    fetcher = fpmms_fetcher()
    for _ in tqdm(fetcher, unit="fpmms", unit_scale=BATCH_SIZE):
        batch = fetcher.send(latest_id)
        if len(batch) == 0:
            break

        latest_id = batch[-1].get(ID_FIELD, "")
        if latest_id == "":
            raise ValueError(f"Unexpected data format retrieved: {batch}")

        fpmms.extend(batch)

    return pd.DataFrame(fpmms)


def get_answer(fpmm: pd.Series) -> str:
    """Get an answer from its index, using Series of an FPMM."""
    return fpmm[QUESTION_FIELD][OUTCOMES_FIELD][fpmm[ANSWER_FIELD]]


def transform_fpmms(fpmms: pd.DataFrame) -> pd.DataFrame:
    """Transform an FPMMS dataframe."""
    transformed = fpmms.dropna()
    transformed = transformed.drop_duplicates([ID_FIELD])
    transformed = transformed.loc[transformed[ANSWER_FIELD] != MAX_UINT_HEX]
    transformed.loc[:, ANSWER_FIELD] = (
        transformed[ANSWER_FIELD].str.slice(-1).astype(int)
    )
    transformed.loc[:, ANSWER_FIELD] = transformed.apply(get_answer, axis=1)
    transformed = transformed.drop(columns=[QUESTION_FIELD])

    return transformed


def etl(filename: Optional[str] = None) -> pd.DataFrame:
    """Fetch, process, store and return the markets as a Dataframe."""
    if os.path.exists(filename):
        old_fpmms = pd.read_csv(filename)
        latest_id = old_fpmms.tail(n=1)["id"].values[0]
    else:
        old_fpmms = None
        latest_id = ""

    fpmms = fetch_fpmms(latest_id)

    n_markets = len(fpmms.index)
    print(f"Found {n_markets} new markets.")

    if n_markets == 0:
        return pd.read_csv(filename)

    fpmms = transform_fpmms(fpmms)

    if old_fpmms is not None:
        fpmms = old_fpmms.append(fpmms)

    if filename:
        fpmms.to_csv(filename, index=False)

    return fpmms


if __name__ == "__main__":
    etl(DEFAULT_FILENAME)
