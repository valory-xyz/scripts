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

"""ETL configurations."""

# Define a dictionary with the data types for each column of the historical data.
HIST_DTYPES = {
    "createdAtBlockNumber": int,
    "createdAtTimestamp": int,
    "id": str,
    "liquidityProviderCount": int,
    "reserve0": float,
    "reserve1": float,
    "reserveETH": float,
    "reserveUSD": float,
    "token0Price": float,
    "token1Price": float,
    "totalSupply": float,
    "trackedReserveETH": float,
    "untrackedVolumeUSD": float,
    "txCount": int,
    "volumeToken0": float,
    "volumeToken1": float,
    "volumeUSD": float,
    "for_timestamp": int,
    "block_number": int,
    "block_timestamp": int,
    "eth_price": float,
    "token0": object,
    "token1": object,
}
