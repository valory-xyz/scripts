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

"""Tools for ETL."""

from typing import Optional, Iterator

from config.general import DAY_IN_UNIX, HOUR_IN_UNIX, MINUTE_IN_UNIX


def gen_unix_timestamps(start: int, interval_in_unix: int, end: Optional[int] = None) -> Iterator[int]:
    """Generate the Unix timestamps from start to end with the given interval.

    :param start: the start date for the generated timestamps.
    :param interval_in_unix: the interval to use in order to generate the timestamps.
    :param end: the end date for the generated timestamps.
    :yields: the UNIX timestamps.
    """
    for timestamp in range(start, end, interval_in_unix):
        yield timestamp


def interval_to_unix(interval: str) -> int:
    """Calculate the interval in Unix.

    :param interval: the interval to use in order to generate the timestamps.
        Can be one of:
           * day
           * hour
           * minute
    :return: the interval in Unix.
    """
    if interval == "day":
        return DAY_IN_UNIX
    elif interval == "hour":
        return HOUR_IN_UNIX
    elif interval == "minute":
        return MINUTE_IN_UNIX
    else:
        raise ValueError(f"Unrecognized interval `{interval}` given. "
                         "Interval can be one of: {`day`, `hour`, `minute`}.")
