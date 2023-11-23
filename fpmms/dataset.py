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

"""
Combines the data from the markets and tools dataframes.
To achieve this, the script requires the filenames of the markets and tools data to be provided as positional arguments.
It assumes the correct order of these filenames for simplicity of implementation; if not, the script will fail.
"""

import sys
import warnings
from typing import Tuple, Optional, Union

import pandas as pd

from markets import etl as markets_etl, TITLE_FIELD, DEFAULT_FILENAME as MARKETS_FILENAME
from tools import etl as tools_etl, PROMPT_FIELD, DEFAULT_FILENAME as TOOLS_FILENAME

DEFAULT_FILENAME = "dataset.csv"


def parse_args() -> Union[str, Tuple[str, str]]:
    """Parse the arguments and return the markets and tools filenames."""
    if len(sys.argv) == 2:
        msg = "No filenames were provided. Fetching all the information from scratch..."
        print(msg)
        return sys.argv[1]
    if len(sys.argv) != 3:
        err = "Expected the paths to the markets and the tools as positional arguments."
        raise ValueError(err)
    return sys.argv[1], sys.argv[2]


def generate(filename: Optional[str] = None) -> pd.DataFrame:
    """Generate the dataset."""
    args = parse_args()

    if isinstance(args, str):
        dfs = markets_etl(MARKETS_FILENAME), tools_etl(args, TOOLS_FILENAME)
    else:
        dfs = tuple(pd.read_csv(filename) for filename in args)

    # extract the titles from the prompts
    fpmms, tools = dfs
    for market_idx in fpmms.index:
        fpmms_title = fpmms.loc[market_idx, TITLE_FIELD]

        # filter warnings to avoid getting:
        # `UserWarning: This pattern has match groups. To actually get the groups, use str.extract.`
        # as some questions might contain something that looks like a regex
        with warnings.catch_warnings():
            warnings.simplefilter(action="ignore", category=UserWarning)
            prompt_contains_title = tools[PROMPT_FIELD].str.contains(fpmms_title)

        tools.loc[prompt_contains_title, TITLE_FIELD] = fpmms_title

    dataset = pd.merge(*dfs, on=TITLE_FIELD)

    if filename:
        dataset.to_csv(filename, index=False)

    return dataset


if __name__ == "__main__":
    generate(DEFAULT_FILENAME)
