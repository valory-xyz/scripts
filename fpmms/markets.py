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

from string import Template
from typing import Optional, Generator

import pandas as pd
import requests
from tqdm import tqdm

from etl import SubgraphResponseType, ResponseItemType
from etl.fetch import hacky_retry

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
          isPendingArbitration: false
        },
        orderBy: creationTimestamp
        orderDirection: desc
        first: ${first}
        skip: ${skip}
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
        total_fetched = yield
        fpmms_query = FPMMS_QUERY.substitute(
            creator=CREATOR,
            fpmms_field=FPMMS_FIELD,
            first=BATCH_SIZE,
            skip=total_fetched,
            id_field=ID_FIELD,
            answer_field=ANSWER_FIELD,
            question_field=QUESTION_FIELD,
            outcomes_field=OUTCOMES_FIELD,
            title_field=TITLE_FIELD,
        )
        yield query_subgraph(OMEN_SUBGRAPH, fpmms_query, FPMMS_FIELD)


def fetch_fpmms() -> pd.DataFrame:
    """Fetch all the fpmms of the creator."""
    fpmms = []
    fetcher = fpmms_fetcher()
    for _ in tqdm(fetcher, unit="fpmms", unit_scale=BATCH_SIZE):
        batch = fetcher.send(len(fpmms))
        if len(batch) == 0:
            break
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
    fpmms = fetch_fpmms()
    fpmms = transform_fpmms(fpmms)

    if filename:
        fpmms.to_csv(filename, index=False)

    return fpmms


if __name__ == "__main__":
    etl(DEFAULT_FILENAME)
