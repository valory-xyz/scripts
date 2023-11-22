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

import json
import sys
import time
from dataclasses import dataclass
from enum import Enum
from typing import Optional, List, Dict, Any

import pandas as pd
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
LATEST_BLOCK: Optional[int] = None
LATEST_BLOCK_NAME: BlockParams = "latest"
BLOCK_DATA_NUMBER = "number"
BLOCKS_CHUNK_SIZE = 20_000
EVENT_ARGUMENTS = "args"
DATA = "data"
REQUEST_ID = "requestId"
REQUEST_SENDER = "sender"
PROMPT_FIELD = "prompt"
CID_PREFIX = "f01701220"
HTTP = "http://"
HTTPS = HTTP[:4] + "s" + HTTP[4:]
IPFS_ADDRESS = f"{HTTPS}gateway.autonolas.tech/ipfs/"
IPFS_LINKS_SERIES_NAME = "ipfs_links"
N_RETRIES = 3
BACKOFF_FACTOR = 1
STATUS_FORCELIST = [500, 502, 503, 504]
DEFAULT_FILENAME = "tools.csv"
SLEEP = 0.5
N_RPC_RETRIES = 100
RPC_POLL_INTERVAL = 0.05
IPFS_POLL_INTERVAL = 0.1


class MechEventName(Enum):
    """The mech's event names."""

    REQUEST = "Request"
    DELIVER = "Deliver"


@dataclass
class MechEvent:
    """A mech's on-chain event representation."""

    requestId: int
    data: bytes
    sender: str

    def _ipfs_link(self) -> Optional[str]:
        """Get the ipfs link for the data."""
        return f"{IPFS_ADDRESS}{CID_PREFIX}{self.data.hex()}"

    @property
    def ipfs_request_link(self) -> Optional[str]:
        """Get the IPFS link for the request."""
        return self._ipfs_link()

    @property
    def ipfs_deliver_link(self) -> Optional[str]:
        """Get the IPFS link for the deliver."""
        if self.requestId is None:
            return None
        return f"{self._ipfs_link()}/{self.requestId}"

    def ipfs_link(self, event_name: MechEventName) -> Optional[str]:
        """Get the ipfs link based on the event."""
        if event_name == MechEventName.REQUEST:
            return self.ipfs_request_link
        if event_name == MechEventName.DELIVER:
            return self.ipfs_deliver_link
        return None


@dataclass(init=False)
class MechRequest:
    """A structure for a request to a mech."""

    requestId: Optional[int]
    prompt: Optional[str]
    tool: Optional[str]
    nonce: Optional[str]

    def __init__(self, **kwargs: Any) -> None:
        """Initialize the request ignoring extra keys."""
        self.requestId = int(kwargs.pop(REQUEST_ID, 0))
        self.prompt = kwargs.pop(PROMPT_FIELD, None)
        self.tool = kwargs.pop("tool", None)
        self.nonce = kwargs.pop("nonce", None)


@dataclass(init=False)
class PredictionResponse:
    """A response of a prediction."""

    p_yes: float
    p_no: float
    confidence: float
    info_utility: float
    vote: Optional[str]
    win_probability: Optional[float]

    def __init__(self, **kwargs: Any) -> None:
        """Initialize the mech's prediction ignoring extra keys."""
        self.p_yes = float(kwargs.pop("p_yes"))
        self.p_no = float(kwargs.pop("p_no"))
        self.confidence = float(kwargs.pop("confidence"))
        self.info_utility = float(kwargs.pop("info_utility"))
        self.win_probability = 0

        # all the fields are probabilities; run checks on whether the current prediction response is valid or not.
        probabilities = (
            getattr(self, field) for field in set(self.__annotations__) - {"vote"}
        )
        if (
            any(not (0 <= prob <= 1) for prob in probabilities)
            or self.p_yes + self.p_no != 1
        ):
            raise ValueError("Invalid prediction response initialization.")

        self.vote = self.get_vote()
        self.win_probability = self.get_win_probability()

    def get_vote(self) -> Optional[str]:
        """Return the vote."""
        if self.p_no == self.p_yes:
            return None
        if self.p_no > self.p_yes:
            return "No"
        return "Yes"

    def get_win_probability(self) -> Optional[float]:
        """Return the probability estimation for winning with vote."""
        return max(self.p_no, self.p_yes)


@dataclass(init=False)
class MechResponse:
    """A structure for the response of a mech."""

    requestId: int
    result: Optional[PredictionResponse]
    error: str

    def __init__(self, **kwargs: Any) -> None:
        """Initialize the mech's response ignoring extra keys."""
        self.requestId = int(kwargs.pop(REQUEST_ID, 0))
        self.error = kwargs.pop("error", "Unknown")
        self.result = kwargs.pop("result", None)

        if isinstance(self.result, str):
            self.result = PredictionResponse(**json.loads(self.result))


EVENT_TO_MECH_STRUCT = {
    MechEventName.REQUEST: MechRequest,
    MechEventName.DELIVER: MechResponse,
}


def parse_args() -> str:
    """Parse the arguments and return the RPC."""
    if len(sys.argv) != 2:
        raise ValueError("Expected the RPC as a positional argument.")
    return sys.argv[1]


def read_abi() -> str:
    """Read and return the wxDAI contract's ABI."""
    with open(MECH_ABI_PATH) as abi_file:
        return abi_file.read()


def get_events(w3: Web3, event: str) -> List:
    """Get the delivered events."""
    abi = read_abi()
    contract_instance = w3.eth.contract(address=MECH_CONTRACT_ADDRESS, abi=abi)

    latest_block = LATEST_BLOCK
    if latest_block is None:
        latest_block = w3.eth.get_block(LATEST_BLOCK_NAME)[BLOCK_DATA_NUMBER]

    events = []
    for from_block in tqdm(
        range(EARLIEST_BLOCK, latest_block, BLOCKS_CHUNK_SIZE),
        desc=f"Searching {event} events in block chunks of size {BLOCKS_CHUNK_SIZE}",
        unit="block chunks",
    ):
        events_filter = contract_instance.events[event].build_filter()
        events_filter.fromBlock = from_block
        events_filter.toBlock = min(from_block + BLOCKS_CHUNK_SIZE, latest_block)

        entries = None
        retries = 0
        while entries is None:
            try:
                entries = events_filter.deploy(w3).get_all_entries()
                retries = 0
            except Exception as exc:
                retries += 1
                if retries == N_RPC_RETRIES:
                    tqdm.write(
                        f"Skipping events for blocks {events_filter.fromBlock} - {events_filter.toBlock} "
                        f"as the retries have been exceeded."
                    )
                    break
                sleep = SLEEP * retries
                tqdm.write(
                    f"An error was raised from the RPC: {exc}\n Retrying in {sleep} seconds."
                )
                time.sleep(sleep)

        if entries is None:
            continue

        chunk = list(entries)
        events.extend(chunk)
        time.sleep(RPC_POLL_INTERVAL)

    return events


def parse_events(raw_events: List) -> List[MechEvent]:
    """Parse all the specified MechEvents."""
    parsed_events = []
    for event in raw_events:
        args = event.get(EVENT_ARGUMENTS, {})
        request_id = args.get(REQUEST_ID, 0)
        data = args.get(DATA, b"")
        sender = args.get(REQUEST_SENDER, "")
        parsed_event = MechEvent(request_id, data, sender)
        parsed_events.append(parsed_event)

    return parsed_events


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


def limit_text(text: str, limit: int = 200) -> str:
    """Limit the given text"""
    if len(text) > limit:
        return f"{text[:limit]}..."
    return text


def get_contents(
    session: requests.Session, events: List[MechEvent], event_name: MechEventName
) -> pd.DataFrame:
    """Fetch the tools' responses."""
    contents = []
    for event in tqdm(events, desc=f"Tools' results", unit="results"):
        raw_content = request(session, event.ipfs_link(event_name))
        struct = EVENT_TO_MECH_STRUCT.get(event_name)

        try:
            raw_content[REQUEST_ID] = str(event.requestId)
            mech_response = struct(**raw_content)
        except (ValueError, TypeError, KeyError):
            tqdm.write(f"Could not parse {limit_text(str(raw_content))}")
            continue
        contents.append(mech_response)
        time.sleep(IPFS_POLL_INTERVAL)

    return pd.DataFrame(contents)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Clean up a dataframe, i.e., drop na and duplicates."""
    return df.dropna().drop_duplicates()


def transform_request(contents: pd.DataFrame) -> pd.DataFrame:
    """Transform the requests dataframe."""
    return clean(contents)


def transform_deliver(contents: pd.DataFrame) -> pd.DataFrame:
    """Transform the delivers dataframe."""
    unpacked_result = pd.json_normalize(contents.result)
    contents = pd.concat((contents, unpacked_result), axis=1)
    return clean(contents.drop(columns=["result", "error"]))


def etl(filename: Optional[str] = None) -> pd.DataFrame:
    """Fetch from on-chain events, process, store and return the tools' results on all the questions as a Dataframe."""
    rpc = parse_args()
    w3 = Web3(HTTPProvider(rpc))
    session = create_session()
    event_to_transformer = {
        MechEventName.REQUEST: transform_request,
        MechEventName.DELIVER: transform_deliver,
    }
    event_to_contents = {}
    for event_name, transformer in event_to_transformer.items():
        events = get_events(w3, event_name.value)
        parsed = parse_events(events)
        contents = get_contents(session, parsed, event_name)
        event_to_contents[event_name] = transformer(contents)

    tools = pd.merge(*event_to_contents.values(), on=REQUEST_ID)
    if filename:
        tools.to_csv(filename, index=False)
    return tools


if __name__ == "__main__":
    etl(DEFAULT_FILENAME)
