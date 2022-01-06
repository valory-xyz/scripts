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

"""ETL operations."""
import time

from etl.fetch import get_pairs_hist
from etl.tools import interval_to_unix
from etl.transform import transform_hist_data


def export_transform_store(**kwargs) -> None:
    """Fetch the historical data, transform and store them."""
    interval_in_unix = interval_to_unix(kwargs["interval"])
    end = kwargs["end_date"]
    if end is None:
        end = int(time.time())

    pairs_hist = get_pairs_hist(kwargs["pool_ids"], kwargs["start_date"], interval_in_unix, kwargs["interval"], end)
    pairs_hist = transform_hist_data(pairs_hist)
    pairs_hist.to_csv('hist_data.csv', index=False)
