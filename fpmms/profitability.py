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

import time
import requests
import datetime
import pandas as pd
from collections import defaultdict
from typing import Any, Union
from string import Template
from enum import Enum
from tqdm import tqdm
import numpy as np


IRRELEVANT_TOOLS = [
    "openai-text-davinci-002",
    "openai-text-davinci-003",
    "openai-gpt-3.5-turbo",
    "openai-gpt-4",
    "stabilityai-stable-diffusion-v1-5",
    "stabilityai-stable-diffusion-xl-beta-v2-2-2",
    "stabilityai-stable-diffusion-512-v2-1",
    "stabilityai-stable-diffusion-768-v2-1",
    "deepmind-optimization-strong",
    "deepmind-optimization",
]
QUERY_BATCH_SIZE = 1000
DUST_THRESHOLD = 10000000000000
INVALID_ANSWER_HEX = "0xffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff"
INVALID_ANSWER = -1
FPMM_CREATOR = "0x89c5cc945dd550bcffb72fe42bff002429f46fec"
DEFAULT_FROM_DATE = "1970-01-01T00:00:00"
DEFAULT_TO_DATE = "2038-01-19T03:14:07"
DEFAULT_FROM_TIMESTAMP = 0
DEFAULT_TO_TIMESTAMP = 2147483647
WXDAI_CONTRACT_ADDRESS = "0xe91D153E0b41518A2Ce8Dd3D7944Fa863463a97d"
DEFAULT_MECH_FEE = 0.01
DUST_THRESHOLD = 10000000000000


class MarketState(Enum):
    """Market state"""

    OPEN = 1
    PENDING = 2
    FINALIZING = 3
    ARBITRATING = 4
    CLOSED = 5

    def __str__(self) -> str:
        """Prints the market status."""
        return self.name.capitalize()


class MarketAttribute(Enum):
    """Attribute"""

    NUM_TRADES = "Num_trades"
    WINNER_TRADES = "Winner_trades"
    NUM_REDEEMED = "Num_redeemed"
    INVESTMENT = "Investment"
    FEES = "Fees"
    MECH_CALLS = "Mech_calls"
    MECH_FEES = "Mech_fees"
    EARNINGS = "Earnings"
    NET_EARNINGS = "Net_earnings"
    REDEMPTIONS = "Redemptions"
    ROI = "ROI"

    def __str__(self) -> str:
        """Prints the attribute."""
        return self.value

    def __repr__(self) -> str:
        """Prints the attribute representation."""
        return self.name

    @staticmethod
    def argparse(s: str) -> "MarketAttribute":
        """Performs string conversion to MarketAttribute."""
        try:
            return MarketAttribute[s.upper()]
        except KeyError as e:
            raise ValueError(f"Invalid MarketAttribute: {s}") from e


ALL_TRADES_STATS_DF_COLS = [
    "trader_address",
    "trade_id",
    "creation_timestamp",
    "title",
    "market_status",
    "collateral_amount",
    "outcome_index",
    "trade_fee_amount",
    "outcomes_tokens_traded",
    "current_answer",
    "is_invalid",
    "winning_trade",
    "earnings",
    "redeemed",
    "redeemed_amount",
    "num_mech_calls",
    "mech_fee_amount",
    "net_earnings",
    "roi",
]

SUMMARY_STATS_DF_COLS = [
    "trader_address",
    "num_trades",
    "num_winning_trades",
    "num_redeemed",
    "total_investment",
    "total_trade_fees",
    "num_mech_calls",
    "total_mech_fees",
    "total_earnings",
    "total_redeemed_amount",
    "total_net_earnings",
    "total_net_earnings_wo_mech_fees",
    "total_roi",
    "total_roi_wo_mech_fees",
    "mean_mech_calls_per_trade",
    "mean_mech_fee_amount_per_trade",
]
headers = {
    "Accept": "application/json, multipart/mixed",
    "Content-Type": "application/json",
}


omen_xdai_trades_query = Template(
    """
    {
        fpmmTrades(
            where: {
                type: Buy,
                fpmm_: {
                    creator: "${fpmm_creator}"
                    creationTimestamp_gte: "${fpmm_creationTimestamp_gte}",
                    creationTimestamp_lt: "${fpmm_creationTimestamp_lte}"
                },
                creationTimestamp_gte: "${creationTimestamp_gte}",
                creationTimestamp_lte: "${creationTimestamp_lte}"
                id_gt: "${id_gt}"
            }
            first: ${first}
            orderBy: id
            orderDirection: asc
        ) {
            id
            title
            collateralToken
            outcomeTokenMarginalPrice
            oldOutcomeTokenMarginalPrice
            type
            creator {
                id
            }
            creationTimestamp
            collateralAmount
            collateralAmountUSD
            feeAmount
            outcomeIndex
            outcomeTokensTraded
            transactionHash
            fpmm {
                id
                outcomes
                title
                answerFinalizedTimestamp
                currentAnswer
                isPendingArbitration
                arbitrationOccurred
                openingTimestamp
                condition {
                    id
                }
            }
        }
    }
    """
)


conditional_tokens_gc_user_query = Template(
    """
    {
        user(id: "${id}") {
            userPositions(
                first: ${first}
                where: {
                    id_gt: "${userPositions_id_gt}"
                }
                orderBy: id
            ) {
                balance
                id
                position {
                    id
                    conditionIds
                }
                totalBalance
                wrappedBalance
            }
        }
    }
    """
)


def _to_content(q: str) -> dict[str, Any]:
    """Convert the given query string to payload content, i.e., add it under a `queries` key and convert it to bytes."""
    finalized_query = {
        "query": q,
        "variables": None,
        "extensions": {"headers": None},
    }
    return finalized_query


def _query_omen_xdai_subgraph(
    from_timestamp: float,
    to_timestamp: float,
    fpmm_from_timestamp: float,
    fpmm_to_timestamp: float,
) -> dict[str, Any]:
    """Query the subgraph."""
    url = "https://api.thegraph.com/subgraphs/name/protofire/omen-xdai"

    grouped_results = defaultdict(list)
    id_gt = ""

    while True:
        query = omen_xdai_trades_query.substitute(
            fpmm_creator=FPMM_CREATOR.lower(),
            creationTimestamp_gte=int(from_timestamp),
            creationTimestamp_lte=int(to_timestamp),
            fpmm_creationTimestamp_gte=int(fpmm_from_timestamp),
            fpmm_creationTimestamp_lte=int(fpmm_to_timestamp),
            first=QUERY_BATCH_SIZE,
            id_gt=id_gt,
        )
        content_json = _to_content(query)
        res = requests.post(url, headers=headers, json=content_json)
        result_json = res.json()
        user_trades = result_json.get("data", {}).get("fpmmTrades", [])

        if not user_trades:
            break

        for trade in user_trades:
            fpmm_id = trade.get("fpmm", {}).get("id")
            grouped_results[fpmm_id].append(trade)

        id_gt = user_trades[len(user_trades) - 1]["id"]

    all_results = {
        "data": {
            "fpmmTrades": [
                trade
                for trades_list in grouped_results.values()
                for trade in trades_list
            ]
        }
    }

    return all_results


def _query_conditional_tokens_gc_subgraph(creator: str) -> dict[str, Any]:
    """Query the subgraph."""
    url = "https://api.thegraph.com/subgraphs/name/gnosis/conditional-tokens-gc"

    all_results: dict[str, Any] = {"data": {"user": {"userPositions": []}}}
    userPositions_id_gt = ""
    while True:
        query = conditional_tokens_gc_user_query.substitute(
            id=creator.lower(),
            first=QUERY_BATCH_SIZE,
            userPositions_id_gt=userPositions_id_gt,
        )
        content_json = {"query": query}
        res = requests.post(url, headers=headers, json=content_json)
        result_json = res.json()
        user_data = result_json.get("data", {}).get("user", {})

        if not user_data:
            break

        user_positions = user_data.get("userPositions", [])

        if user_positions:
            all_results["data"]["user"]["userPositions"].extend(user_positions)
            userPositions_id_gt = user_positions[len(user_positions) - 1]["id"]
        else:
            break

    if len(all_results["data"]["user"]["userPositions"]) == 0:
        return {"data": {"user": None}}

    return all_results


def convert_hex_to_int(x: Union[str, float]) -> Union[int, float]:
    """Convert hex to int"""
    if isinstance(x, float):
        return np.nan
    elif isinstance(x, str):
        if x == INVALID_ANSWER_HEX:
            return -1
        else: 
            return int(x, 16)
        
def wei_to_unit(wei: int) -> float:
    """Converts wei to currency unit."""
    return wei / 10**18


def _is_redeemed(user_json: dict[str, Any], fpmmTrade: dict[str, Any]) -> bool:
    """Returns whether the user has redeemed the position."""
    user_positions = user_json["data"]["user"]["userPositions"]
    outcomes_tokens_traded = int(fpmmTrade["outcomeTokensTraded"])
    condition_id = fpmmTrade["fpmm.condition.id"]

    for position in user_positions:
        position_condition_ids = position["position"]["conditionIds"]
        balance = int(position["balance"])

        if condition_id in position_condition_ids and balance == outcomes_tokens_traded:
            return False

    for position in user_positions:
        position_condition_ids = position["position"]["conditionIds"]
        balance = int(position["balance"])

        if condition_id in position_condition_ids and balance == 0:
            return True

    return False


def create_fpmmTrades(rpc: str):
    """Create fpmmTrades for all trades."""
    trades_json = _query_omen_xdai_subgraph(
        from_timestamp=DEFAULT_FROM_TIMESTAMP,
        to_timestamp=DEFAULT_TO_TIMESTAMP,
        fpmm_from_timestamp=DEFAULT_FROM_TIMESTAMP,
        fpmm_to_timestamp=DEFAULT_TO_TIMESTAMP,
    )

    # convert to dataframe
    df = pd.DataFrame(trades_json["data"]["fpmmTrades"])

    # convert creator to address
    df["creator"] = df["creator"].apply(lambda x: x["id"])

    # normalize fpmm column
    fpmm = pd.json_normalize(df["fpmm"])
    fpmm.columns = [f"fpmm.{col}" for col in fpmm.columns]
    df = pd.concat([df, fpmm], axis=1)

    # drop fpmm column
    df.drop(["fpmm"], axis=1, inplace=True)

    # change creator to creator_address
    df.rename(columns={"creator": "trader_address"}, inplace=True)

    # save to csv
    df.to_csv("fpmmTrades.csv", index=False)

    return df


def prepare_profitalibity_data(rpc: str):
    """Prepare data for profitalibity analysis."""

    # Check if tools.py is in the same directory
    try:
        # load tools.csv
        tools = pd.read_csv("tools.csv")

        # make sure creator_address is in the columns
        assert "trader_address" in tools.columns, "trader_address column not found"

        # lowercase and strip creator_address
        tools["trader_address"] = tools["trader_address"].str.lower().str.strip()

        # drop duplicates
        tools.drop_duplicates(inplace=True)

        print("tools.csv loaded")
    except FileNotFoundError:
        print("tools.csv not found. Please run tools.py first.")
        return

    # Check if fpmmTrades.csv is in the same directory
    try:
        # load fpmmTrades.csv
        fpmmTrades = pd.read_csv("fpmmTrades.csv")
        print("fpmmTrades.csv loaded")
    except FileNotFoundError:
        print("fpmmTrades.csv not found. Creating fpmmTrades.csv...")
        fpmmTrades = create_fpmmTrades(rpc)
        fpmmTrades.to_csv("fpmmTrades.csv", index=False)
        fpmmTrades = pd.read_csv("fpmmTrades.csv")

    # make sure trader_address is in the columns
    assert "trader_address" in fpmmTrades.columns, "trader_address column not found"

    # lowercase and strip creator_address
    fpmmTrades["trader_address"] = (
        fpmmTrades["trader_address"].str.lower().str.strip()
    )


    return fpmmTrades, tools


def determine_market_status(trade, current_answer):
    """Determine the market status of a trade."""
    if current_answer is np.nan and time.time() >= trade["fpmm.openingTimestamp"]:
        return MarketState.PENDING
    elif current_answer == np.nan:
        return MarketState.OPEN
    elif trade["fpmm.isPendingArbitration"]:
        return MarketState.ARBITRATING
    elif time.time() < trade["fpmm.answerFinalizedTimestamp"]:
        return MarketState.FINALIZING
    return MarketState.CLOSED


def analyse_trader(trader_address: str, fpmmTrades: pd.DataFrame, tools: pd.DataFrame) -> pd.DataFrame:
    """Analyse a trader's trades"""
    # Filter trades and tools for the given trader
    trades = fpmmTrades[fpmmTrades["trader_address"] == trader_address]
    tools_usage = tools[tools["trader_address"] == trader_address]

    # Prepare the DataFrame
    trades_df = pd.DataFrame(columns=ALL_TRADES_STATS_DF_COLS)
    if trades.empty:
        return trades_df

    # Fetch user's conditional tokens gc graph
    try:
        user_json = _query_conditional_tokens_gc_subgraph(trader_address)
    except Exception as e:
        print(f"Error fetching user data: {e}")
        return trades_df

    # Iterate over the trades
    for i, trade in tqdm(trades.iterrows(), total=len(trades), desc="Analysing trades"):
        try:
            # Parsing and computing shared values
            creation_timestamp_utc = datetime.datetime.fromtimestamp(trade["creationTimestamp"], tz=datetime.timezone.utc)
            collateral_amount = wei_to_unit(float(trade["collateralAmount"]))
            fee_amount = wei_to_unit(float(trade["feeAmount"]))
            outcome_tokens_traded = wei_to_unit(float(trade["outcomeTokensTraded"]))
            earnings, winner_trade = (0, False)
            redemption = _is_redeemed(user_json, trade)
            current_answer = trade["fpmm.currentAnswer"]

            # Determine market status
            market_status = determine_market_status(trade, current_answer)

            # Skip non-closed markets
            if market_status != MarketState.CLOSED:
                print(f"Skipping trade {i} because market is not closed. Market Status: {market_status}")
                continue
            current_answer = convert_hex_to_int(current_answer)
            
            # Compute invalidity
            is_invalid = current_answer == INVALID_ANSWER

            # Compute earnings and winner trade status
            if is_invalid:
                earnings = collateral_amount
                winner_trade = False
            elif trade["outcomeIndex"] == current_answer:
                earnings = outcome_tokens_traded
                winner_trade = True

            # Compute mech calls
            num_mech_calls = tools_usage["prompt_request"].apply(lambda x: trade["title"] in x).sum()
            net_earnings = earnings - fee_amount - (num_mech_calls * DEFAULT_MECH_FEE) - collateral_amount

            # Assign values to DataFrame
            trades_df.loc[i] = {
                "trader_address": trader_address,
                "trade_id": trade["id"],
                "market_status": market_status.name,
                "creation_timestamp": creation_timestamp_utc,
                "title": trade["title"],
                "collateral_amount": collateral_amount,
                "outcome_index": trade["outcomeIndex"],
                "trade_fee_amount": fee_amount,
                "outcomes_tokens_traded": outcome_tokens_traded,
                "current_answer": current_answer,
                "is_invalid": is_invalid,
                "winning_trade": winner_trade,
                "earnings": earnings,
                "redeemed": redemption,
                "redeemed_amount": earnings if redemption else 0,
                "num_mech_calls": num_mech_calls,
                "mech_fee_amount": num_mech_calls * DEFAULT_MECH_FEE,
                "net_earnings": net_earnings,
                "roi": net_earnings / collateral_amount
            }

        except Exception as e:
            print(f"Error processing trade {i}: {e}")
            continue

    return trades_df


def analyse_all_traders(trades: pd.DataFrame, tools: pd.DataFrame) -> pd.DataFrame:
    """Analyse all creators."""
    all_traders = []
    for trader in tqdm(
        trades["trader_address"].unique(),
        total=len(trades["trader_address"].unique()),
        desc="Analysing creators",
    ):
        all_traders.append(
            analyse_trader(trader, trades, tools)
        )

    # concat all creators
    all_creators_df = pd.concat(all_traders)

    return all_creators_df


def summary_analyse(df):
    """Summarise profitability analysis."""
    # Ensure DataFrame is not empty
    if df.empty:
        return pd.DataFrame(columns=SUMMARY_STATS_DF_COLS)

    # Group by trader_address
    grouped = df.groupby('trader_address')

    # Create summary DataFrame
    summary_df = grouped.agg(
        num_trades=('trader_address', 'size'),
        num_winning_trades=('winning_trade', lambda x: float((x).sum())),
        num_redeemed=('redeemed', lambda x: float(x.sum())),
        total_investment=('collateral_amount', 'sum'),
        total_trade_fees=('trade_fee_amount', 'sum'),
        num_mech_calls=('num_mech_calls', 'sum'),
        total_mech_fees=('mech_fee_amount', 'sum'),
        total_earnings=('earnings', 'sum'),
        total_redeemed_amount=('redeemed_amount', 'sum'),
        total_net_earnings=('net_earnings', 'sum')
    )

    # Calculating additional columns
    summary_df['total_roi'] = summary_df['total_net_earnings'] / summary_df['total_investment']
    summary_df['mean_mech_calls_per_trade'] = summary_df['num_mech_calls'] / summary_df['num_trades']
    summary_df['mean_mech_fee_amount_per_trade'] = summary_df['total_mech_fees'] / summary_df['num_trades']
    summary_df['total_net_earnings_wo_mech_fees'] = summary_df['total_net_earnings'] + summary_df['total_mech_fees']
    summary_df['total_roi_wo_mech_fees'] = summary_df['total_net_earnings_wo_mech_fees'] / summary_df['total_investment']

    # Resetting index to include trader_address
    summary_df.reset_index(inplace=True)

    return summary_df


def run_profitability_analysis(rpc):
    """Create all trades analysis."""

    # load dfs from csv for analysis
    print("Preparing data...")
    fpmmTrades, tools = prepare_profitalibity_data(rpc)

    # all trades profitability df
    print("Analysing trades...")
    all_trades_df = analyse_all_traders(fpmmTrades, tools)

    # summarize profitability df
    print("Summarising trades...")
    summary_df = summary_analyse(all_trades_df)

    # save to csv
    all_trades_df.to_csv("all_trades_profitability.csv", index=False)
    summary_df.to_csv("summary_profitability.csv", index=False)

    print("Done!")

    return all_trades_df, summary_df


if __name__ == "__main__":
    rpc = "https://lb.nodies.app/v1/406d8dcc043f4cb3959ed7d6673d311a"
    run_profitability_analysis(rpc)