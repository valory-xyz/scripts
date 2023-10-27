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

"""CLI functionality."""

from argparse import ArgumentParser


def create_parser() -> ArgumentParser:
    """Creates the script's argument parser."""
    parser = ArgumentParser(description='A tool that provides summative statistics for SpookySwap.')
    parser.add_argument('start_date', type=int, help="The start date in Unix format.")
    parser.add_argument('-e', '--end_date', type=int, default=None,
                        help="The end date in Unix format. Defaults to now.")
    parser.add_argument('pool_ids', nargs="+", type=str, help="The pool ids for which the statistics will be fetched.")
    parser.add_argument('interval', type=int, help="The interval to use to fetch the historical data in Unix.")

    return parser
