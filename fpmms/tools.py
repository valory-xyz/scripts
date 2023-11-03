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

import csv
import sys
from dataclasses import dataclass
from typing import Optional, List, Dict

import requests
from eth_utils import to_checksum_address
from requests.adapters import HTTPAdapter
from tqdm import tqdm
from urllib3 import Retry
from web3 import Web3, HTTPProvider
from web3.types import BlockParams

MECH_CONTRACT_ADDRESS = to_checksum_address(
    "0xff82123dfb52ab75c417195c5fdb87630145ae81"
)
MECH_ABI_PATH = "./contracts/mech_abi.json"
# this is when the creator had its first tx ever
EARLIEST_BLOCK = 28911547
# optionally set the latest block to stop searching for the delivered events
LATEST_BLOCK: Optional[int] = 28931547
LATEST_BLOCK_NAME: BlockParams = "latest"
BLOCK_DATA_NUMBER = "number"
BLOCKS_CHUNK_SIZE = 5_000
EVENT_ARGUMENTS = "args"
DELIVER_REQUEST_ID = "requestId"
DELIVER_DATA = "data"
CID_PREFIX = "f01701220"
HTTP = "http://"
HTTPS = HTTP[:4] + "s" + HTTP[4:]
IPFS_ADDRESS = f"{HTTPS}gateway.autonolas.tech/ipfs/"
IPFS_LINKS_SERIES_NAME = "ipfs_links"
N_RETRIES = 3
BACKOFF_FACTOR = 1
STATUS_FORCELIST = [500, 502, 503, 504]
PROMPT_FIELD = "prompt"


def parse_args() -> str:
    """Parse the arguments and return the RPC."""
    if len(sys.argv) != 2:
        raise ValueError("Expected the RPC as a positional argument.")
    return sys.argv[1]


def read_abi() -> str:
    """Read and return the wxDAI contract's ABI."""
    with open(MECH_ABI_PATH) as abi_file:
        return abi_file.read()


@dataclass
class MechDeliver:
    """A mech's on-chain response representation."""

    request_id: int
    data: bytes

    @property
    def ipfs_link(self) -> Optional[str]:
        """Get the ipfs link."""
        if self.request_id is None:
            return None
        return f"{IPFS_ADDRESS}{CID_PREFIX}{self.data.hex()}/{self.request_id}"


def get_delivered_events(w3: Web3) -> List:
    """Get the delivered events."""
    abi = read_abi()
    contract_instance = w3.eth.contract(address=MECH_CONTRACT_ADDRESS, abi=abi)

    latest_block = LATEST_BLOCK
    if latest_block is None:
        latest_block = w3.eth.get_block(LATEST_BLOCK_NAME)[BLOCK_DATA_NUMBER]

    events = []
    for from_block in tqdm(
        range(EARLIEST_BLOCK, latest_block, BLOCKS_CHUNK_SIZE),
        desc=f"Searching delivered tasks in block chunks of size {BLOCKS_CHUNK_SIZE}",
        unit="block chunks",
    ):
        deliver_filter = contract_instance.events.Deliver.build_filter()
        deliver_filter.fromBlock = EARLIEST_BLOCK
        deliver_filter.toBlock = min(from_block + BLOCKS_CHUNK_SIZE, latest_block)
        chunk = list(deliver_filter.deploy(w3).get_all_entries())
        events.extend(chunk)

    return events


def parse_deliver_events(events: List) -> List[MechDeliver]:
    """Get all the mech delivers from the delivered events."""
    delivers = []
    for delivered_event in events:
        args = delivered_event.get(EVENT_ARGUMENTS, {})
        request_id = args.get(DELIVER_REQUEST_ID, None)
        data = args.get(DELIVER_DATA, b"")
        deliver = MechDeliver(request_id=request_id, data=data)
        delivers.append(deliver)

    return delivers


def create_session() -> requests.Session:
    """Create a session with a retry strategy."""
    session = requests.Session()
    retry_strategy = Retry(
        total=N_RETRIES + 1,
        backoff_factor=BACKOFF_FACTOR,
        status_forcelist=STATUS_FORCELIST,
    )
    adapter = HTTPAdapter(max_retries=retry_strategy)
    for protocol in (HTTP, HTTPS):
        session.mount(protocol, adapter)

    return session


def request(session: requests.Session, url: str) -> Dict[str, str]:
    """Perform a request with a session."""
    try:
        response = session.get(url)
        response.raise_for_status()
    except requests.exceptions.HTTPError as exc:
        print("HTTP error occurred:", exc)
    except Exception as exc:
        print("Unexpected error occurred:", exc)
    else:
        return response.json()


def fetch_responses(ipfs_links: List[str]):
    """Fetch the tools' responses."""
    session = create_session()

    responses = []
    for link in tqdm(ipfs_links, desc=f"Tools' results", unit="results"):
        responses.append(request(session, link))

    return responses


def etl() -> None:
    """Fetch from on-chain events, process, store and return the tools' results on all the questions as a Dataframe."""
    rpc = parse_args()
    w3 = Web3(HTTPProvider(rpc))

    # TODO get request events
    # TODO get delivered events using the request ids of the fetched request events
    delivered_events = get_delivered_events(w3)
    mech_delivers = parse_deliver_events(delivered_events)
    ipfs_links = [deliver.ipfs_link for deliver in mech_delivers]

    # store progress so far.
    filename = f"{IPFS_LINKS_SERIES_NAME}.csv"
    with open(filename, "w") as csv_file:
        write = csv.writer(csv_file)
        write.writerow(ipfs_links)

    print(fetch_responses(ipfs_links))

    # TODO store results
    # TODO on a separate file, map the questions from the request events and the response's outcome from the delivered events with the fpmms dataset obtained from `markets.py`


if __name__ == "__main__":
    etl()
