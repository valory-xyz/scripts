"""Tools for ETL."""
import time
from typing import Optional, Iterator

from config import DAY_IN_UNIX, HOUR_IN_UNIX, MINUTE_IN_UNIX


def gen_unix_timestamps(start: int, interval: str, end: Optional[int] = None) -> Iterator[int]:
    """Generate the Unix timestamps from start to end with the given interval.

    :param start: the start date for the generated timestamps.
    :param interval: the interval to use in order to generate the timestamps.
        Can be one of:
           * day
           * hour
           * minute
    :param end: the end date for the generated timestamps.
    :yields: the UNIX timestamps.
    """
    if end is None:
        end = time.time()

    if interval == "day":
        interval_in_unix = DAY_IN_UNIX
    elif interval == "hour":
        interval_in_unix = HOUR_IN_UNIX
    elif interval == "minute":
        interval_in_unix = MINUTE_IN_UNIX
    else:
        raise ValueError(f"Unrecognized interval `{interval}` given. "
                         "Interval can be one of: {`day`, `hour`, `minute`}.")

    for timestamp in range(start, end, interval_in_unix):
        yield timestamp
