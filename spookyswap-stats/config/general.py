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

"""General configurations."""

_subgraph_base_url = "https://api.thegraph.com/subgraphs/name/"
SPOOKY_SUBGRAPH_URL = _subgraph_base_url + "eerieeight/spookyswap"
FANTOM_BLOCKS_SUBGRAPH_URL = _subgraph_base_url + "matthewlilley/fantom-blocks"
BUNDLE_ID = 1
MINUTE_IN_UNIX = 60
HOUR_IN_UNIX = 60 * MINUTE_IN_UNIX
DAY_IN_UNIX = 24 * HOUR_IN_UNIX
