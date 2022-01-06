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

"""ETL related operations."""

from typing import Optional

import pandas as pd

from config.etl import HIST_DTYPES
from etl import ResponseItemType


def calc_change(volume_usd: pd.Series) -> pd.Series:
    """Calculate the change between two days. The series argument needs to be sorted by date!

    :param volume_usd: a series with volumes, sorted by date.
    :return: a new series with the current day's change for each date of the given `volume_usd` series.
    """
    return volume_usd - volume_usd.shift(1)


def calc_apy(x: pd.Series) -> Optional[float]:
    """Calculates the APY for a given pool's info.

    :param x: a series with historical data.
    :return: the daily APY.
    """
    res = None

    if x["updatedReserveUSD"]:
        res = (x["current_change"] * 0.002 * 365 * 100) / x["updatedReserveUSD"]

    return res


def transform_hist_data(pairs_hist_raw: ResponseItemType) -> pd.DataFrame:
    """Transform pairs' history into a dataframe and add extra fields.

    :param pairs_hist_raw: the pairs historical data non-transformed.
    :return: a dataframe with the given historical data, containing extra fields. These are:
         * [token0_id, token0_name, token0_symbol]: split from `token0`.
         * [token1_id, token1_name, token1_symbol]: split from `token1`.
         * pairName: the current's pair's name, which is derived by: 'token0_name - token1_name'.
         * updatedVolumeUSD: the tracked volume USD, but if it is 0, then the untracked volume USD.
         * updatedReserveUSD: the tracked reserve USD, but if it is 0, then the untracked reserve USD.
         * current_change: the current day's change in volume USD.
         * APY: the APY.

         The dataframe is also sorted by the block's timestamp and the pair's tokens' names,
         and the entries for which the APY cannot be calculated are being dropped.
    """
    # Convert history to a dataframe.
    pairs_hist = pd.DataFrame(pairs_hist_raw).astype(HIST_DTYPES)

    # Split the dictionary-like token cols.
    token_cols = ["token0", "token1"]
    for token_col in token_cols:
        pairs_hist[f"{token_col}_id"] = (
            pairs_hist[token_col].apply(lambda x: x["id"]).astype(str)
        )
        pairs_hist[f"{token_col}_name"] = (
            pairs_hist[token_col].apply(lambda x: x["name"]).astype(str)
        )
        pairs_hist[f"{token_col}_symbol"] = (
            pairs_hist[token_col].apply(lambda x: x["symbol"]).astype(str)
        )
    # Drop the original dictionary-like token cols.
    pairs_hist.drop(columns=token_cols, inplace=True)

    # Create pair's name.
    pairs_hist["pairName"] = (
        pairs_hist["token0_name"] + " - " + pairs_hist["token1_name"]
    )

    # Create a volume USD and a reserve USD taking the untracked amount into consideration as well.
    pairs_hist["updatedVolumeUSD"] = pairs_hist["volumeUSD"]
    pairs_hist["updatedReserveUSD"] = (
        pairs_hist["trackedReserveETH"] * pairs_hist["eth_price"]
    )

    # Create a mask for all the untracked cases.
    not_tracked_m = pairs_hist["updatedVolumeUSD"] == 0

    # Keep the untracked amount for the untracked cases.
    pairs_hist.loc[not_tracked_m, "updatedVolumeUSD"] = pairs_hist.loc[
        not_tracked_m, "untrackedVolumeUSD"
    ]
    pairs_hist.loc[not_tracked_m, "updatedReserveUSD"] = pairs_hist.loc[
        not_tracked_m, "reserveUSD"
    ]

    # Calculate the current change of volume in USD.
    pairs_hist["current_change"] = pairs_hist.groupby("id")[
        "updatedVolumeUSD"
    ].transform(calc_change)
    # Drop NaN values (essentially, this is the first day's `current_change`, because we cannot calculate it).
    pairs_hist.dropna(inplace=True)

    # Calculate APY.
    pairs_hist["APY"] = pairs_hist.apply(calc_apy, axis=1)
    # Drop rows with NaN APY values.
    pairs_hist.dropna(subset=["APY"], inplace=True)

    # Sort the dictionary.
    pairs_hist.sort_values(
        by=["block_timestamp", "token0_symbol", "token1_symbol"],
        ascending=True,
        inplace=True,
    )

    return pairs_hist