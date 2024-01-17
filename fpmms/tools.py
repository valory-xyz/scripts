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
import os.path
import re
import sys
import time
import random
from dataclasses import dataclass
from enum import Enum
from io import StringIO
from typing import (
    Optional,
    List,
    Dict,
    Any,
    Union,
    Callable,
    Tuple,
)

import pandas as pd
import requests
from json.decoder import JSONDecodeError
from eth_typing import ChecksumAddress
from eth_utils import to_checksum_address
from requests.adapters import HTTPAdapter
from requests.exceptions import (
    ReadTimeout as RequestsReadTimeoutError,
    HTTPError as RequestsHTTPError,
)
from tqdm import tqdm
from urllib3 import Retry
from urllib3.exceptions import (
    ReadTimeoutError as Urllib3ReadTimeoutError,
    HTTPError as Urllib3HTTPError,
)
from web3 import Web3, HTTPProvider
from web3.exceptions import MismatchedABI
from web3.types import BlockParams
from concurrent.futures import ThreadPoolExecutor, as_completed

CONTRACTS_PATH = "contracts"
MECH_TO_INFO = {
    # this block number is when the creator had its first tx ever, and after this mech's creation
    "0xff82123dfb52ab75c417195c5fdb87630145ae81": ("old_mech_abi.json", 28911547),
    # this block number is when this mech was created
    "0x77af31de935740567cf4ff1986d04b2c964a786a": ("new_mech_abi.json", 30776879),
}
# optionally set the latest block to stop searching for the delivered events
LATEST_BLOCK: Optional[int] = None
LATEST_BLOCK_NAME: BlockParams = "latest"
BLOCK_DATA_NUMBER = "number"
BLOCKS_CHUNK_SIZE = 10_000
REDUCE_FACTOR = 0.25
EVENT_ARGUMENTS = "args"
DATA = "data"
REQUEST_ID = "requestId"
REQUEST_ID_FIELD = "request_id"
REQUEST_SENDER = "sender"
PROMPT_FIELD = "prompt"
BLOCK_FIELD = "block"
CID_PREFIX = "f01701220"
HTTP = "http://"
HTTPS = HTTP[:4] + "s" + HTTP[4:]
IPFS_ADDRESS = f"{HTTPS}gateway.autonolas.tech/ipfs/"
IPFS_LINKS_SERIES_NAME = "ipfs_links"
BACKOFF_FACTOR = 1
STATUS_FORCELIST = [404, 500, 502, 503, 504]
DEFAULT_FILENAME = "tools.csv"
RE_RPC_FILTER_ERROR = r"Filter with id: '\d+' does not exist."
ABI_ERROR = "The event signature did not match the provided ABI"
SLEEP = 0.5
HTTP_TIMEOUT = 60
N_IPFS_RETRIES = 2
N_RPC_RETRIES = 100
RPC_POLL_INTERVAL = 0.05
IPFS_POLL_INTERVAL = 0.05
FORMAT_UPDATE_BLOCK_NUMBER = 30411638
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
# this is how frequently we will keep a snapshot of the progress so far in terms of blocks' batches
# for example, the value 1 means that for every `BLOCKS_CHUNK_SIZE` blocks that we search, we also store the snapshot
SNAPSHOT_RATE = 10
NUM_WORKERS = 10
GET_CONTENTS_BATCH_SIZE = 1000


class MechEventName(Enum):
    """The mech's event names."""

    REQUEST = "Request"
    DELIVER = "Deliver"


@dataclass
class MechEvent:
    """A mech's on-chain event representation."""

    for_block: int
    requestId: int
    data: bytes
    sender: str

    def _ipfs_link(self) -> Optional[str]:
        """Get the ipfs link for the data."""
        return f"{IPFS_ADDRESS}{CID_PREFIX}{self.data.hex()}"

    @property
    def ipfs_request_link(self) -> Optional[str]:
        """Get the IPFS link for the request."""
        return f"{self._ipfs_link()}/metadata.json"

    @property
    def ipfs_deliver_link(self) -> Optional[str]:
        """Get the IPFS link for the deliver."""
        if self.requestId is None:
            return None
        return f"{self._ipfs_link()}/{self.requestId}"

    def ipfs_link(self, event_name: MechEventName) -> Optional[str]:
        """Get the ipfs link based on the event."""
        if event_name == MechEventName.REQUEST:
            if self.for_block < FORMAT_UPDATE_BLOCK_NUMBER:
                return self._ipfs_link()
            return self.ipfs_request_link
        if event_name == MechEventName.DELIVER:
            return self.ipfs_deliver_link
        return None


@dataclass(init=False)
class MechRequest:
    """A structure for a request to a mech."""

    request_id: Optional[int]
    request_block: Optional[int]
    prompt_request: Optional[str]
    tool: Optional[str]
    nonce: Optional[str]

    def __init__(self, **kwargs: Any) -> None:
        """Initialize the request ignoring extra keys."""
        self.request_id = int(kwargs.pop(REQUEST_ID, 0))
        self.request_block = int(kwargs.pop(BLOCK_FIELD, 0))
        self.prompt_request = kwargs.pop(PROMPT_FIELD, None)
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
        try:
            self.p_yes = float(kwargs.pop("p_yes"))
            self.p_no = float(kwargs.pop("p_no"))
            self.confidence = float(kwargs.pop("confidence"))
            self.info_utility = float(kwargs.pop("info_utility"))
            self.win_probability = 0

            # Validate probabilities
            probabilities = {
                "p_yes": self.p_yes,
                "p_no": self.p_no,
                "confidence": self.confidence,
                "info_utility": self.info_utility,
            }

            for name, prob in probabilities.items():
                if not 0 <= prob <= 1:
                    raise ValueError(f"{name} probability is out of bounds: {prob}")

            if self.p_yes + self.p_no != 1:
                raise ValueError(
                    f"Sum of p_yes and p_no is not 1: {self.p_yes} + {self.p_no}"
                )

            self.vote = self.get_vote()
            self.win_probability = self.get_win_probability()

        except KeyError as e:
            raise KeyError(f"Missing key in PredictionResponse: {e}")
        except ValueError as e:
            raise ValueError(f"Invalid value in PredictionResponse: {e}")

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

    request_id: int
    deliver_block: Optional[int]
    result: Optional[PredictionResponse]
    error: Optional[str]
    error_message: Optional[str]
    prompt_response: Optional[str]

    def __init__(self, **kwargs: Any) -> None:
        """Initialize the mech's response ignoring extra keys."""
        self.error = kwargs.get("error", None)
        self.request_id = int(kwargs.get(REQUEST_ID, 0))
        self.deliver_block = int(kwargs.get(BLOCK_FIELD, 0))
        self.result = kwargs.get("result", None)
        self.prompt_response = kwargs.get(PROMPT_FIELD, None)

        if self.result != "Invalid response":
            self.error_message = kwargs.get("error_message", None)

            try:
                if isinstance(self.result, str):
                    kwargs = json.loads(self.result)
                    self.result = PredictionResponse(**kwargs)
                    self.error = str(False)

            except JSONDecodeError:
                self.error_message = "Response parsing error"
                self.error = str(True)

            except Exception as e:
                self.error_message = str(e)
                self.error = str(True)

        else:
            self.error_message = "Invalid response from tool"
            self.error = str(True)
            self.result = None


EVENT_TO_MECH_STRUCT = {
    MechEventName.REQUEST: MechRequest,
    MechEventName.DELIVER: MechResponse,
}


def parse_args() -> str:
    """Parse the arguments and return the RPC."""
    if len(sys.argv) != 2:
        raise ValueError("Expected the RPC as a positional argument.")
    return sys.argv[1]


def read_abi(abi_path: str) -> str:
    """Read and return the wxDAI contract's ABI."""
    with open(abi_path) as abi_file:
        return abi_file.read()


def reduce_window(contract_instance, event, from_block, batch_size, latest_block):
    """Dynamically reduce the batch size window."""
    keep_fraction = 1 - REDUCE_FACTOR
    events_filter = contract_instance.events[event].build_filter()
    events_filter.fromBlock = from_block
    batch_size = int(batch_size * keep_fraction)
    events_filter.toBlock = min(from_block + batch_size, latest_block)
    tqdm.write(f"RPC timed out! Resizing batch size to {batch_size}.")
    time.sleep(SLEEP)
    return events_filter, batch_size


def get_events(
    w3: Web3,
    event: str,
    mech_address: ChecksumAddress,
    mech_abi_path: str,
    earliest_block: int,
    latest_block: int,
) -> List:
    """Get the delivered events."""
    abi = read_abi(mech_abi_path)
    contract_instance = w3.eth.contract(address=mech_address, abi=abi)

    events = []
    from_block = earliest_block
    batch_size = BLOCKS_CHUNK_SIZE
    with tqdm(
        total=latest_block - from_block,
        desc=f"Searching {event} events for mech {mech_address}",
        unit="blocks",
    ) as pbar:
        while from_block < latest_block:
            events_filter = contract_instance.events[event].build_filter()
            events_filter.fromBlock = from_block
            events_filter.toBlock = min(from_block + batch_size, latest_block)

            entries = None
            retries = 0
            while entries is None:
                try:
                    entries = events_filter.deploy(w3).get_all_entries()
                    retries = 0
                except (RequestsHTTPError, Urllib3HTTPError) as exc:
                    if "Request Entity Too Large" in exc.args[0]:
                        events_filter, batch_size = reduce_window(
                            contract_instance,
                            event,
                            from_block,
                            batch_size,
                            latest_block,
                        )
                except (Urllib3ReadTimeoutError, RequestsReadTimeoutError):
                    events_filter, batch_size = reduce_window(
                        contract_instance, event, from_block, batch_size, latest_block
                    )
                except Exception as exc:
                    retries += 1
                    if retries == N_RPC_RETRIES:
                        tqdm.write(
                            f"Skipping events for blocks {events_filter.fromBlock} - {events_filter.toBlock} "
                            f"as the retries have been exceeded."
                        )
                        break
                    sleep = SLEEP * retries
                    if (
                        (
                            isinstance(exc, ValueError)
                            and re.match(
                                RE_RPC_FILTER_ERROR, exc.args[0].get("message", "")
                            )
                            is None
                        )
                        and not isinstance(exc, ValueError)
                        and not isinstance(exc, MismatchedABI)
                    ):
                        tqdm.write(
                            f"An error was raised from the RPC: {exc}\n Retrying in {sleep} seconds."
                        )
                    time.sleep(sleep)

            from_block += batch_size
            pbar.update(batch_size)

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
        for_block = event.get("blockNumber", 0)
        args = event.get(EVENT_ARGUMENTS, {})
        request_id = args.get(REQUEST_ID, 0)
        data = args.get(DATA, b"")
        sender = args.get(REQUEST_SENDER, "")
        parsed_event = MechEvent(for_block, request_id, data, sender)
        parsed_events.append(parsed_event)

    return parsed_events


def create_session() -> requests.Session:
    """Create a session with a retry strategy."""
    session = requests.Session()
    retry_strategy = Retry(
        total=N_IPFS_RETRIES + 1,
        backoff_factor=BACKOFF_FACTOR,
        status_forcelist=STATUS_FORCELIST,
    )
    adapter = HTTPAdapter(max_retries=retry_strategy)
    for protocol in (HTTP, HTTPS):
        session.mount(protocol, adapter)

    return session


def request(
    session: requests.Session, url: str, timeout: int = HTTP_TIMEOUT
) -> Optional[requests.Response]:
    """Perform a request with a session."""
    try:
        response = session.get(url, timeout=timeout)
        response.raise_for_status()
    except requests.exceptions.HTTPError as exc:
        tqdm.write(f"HTTP error occurred: {exc}.")
    except Exception as exc:
        tqdm.write(f"Unexpected error occurred: {exc}.")
    else:
        return response
    return None


def limit_text(text: str, limit: int = 200) -> str:
    """Limit the given text"""
    if len(text) > limit:
        return f"{text[:limit]}..."
    return text


def parse_ipfs_response(
    session: requests.Session,
    url: str,
    event: MechEvent,
    event_name: MechEventName,
    response: requests.Response,
) -> Optional[Dict[str, str]]:
    """Parse a response from IPFS."""
    try:
        return response.json()
    except requests.exceptions.JSONDecodeError:
        # this is a workaround because the `metadata.json` file was introduced and removed multiple times
        if event_name == MechEventName.REQUEST and url != event.ipfs_request_link:
            url = event.ipfs_request_link
            response = request(session, url)
            if response is None:
                tqdm.write(f"Skipping {event=}.")
                return None

            try:
                return response.json()
            except requests.exceptions.JSONDecodeError:
                pass

    tqdm.write(f"Failed to parse response into json for {url=}.")
    return None


def parse_ipfs_tools_content(
    raw_content: Dict[str, str], event: MechEvent, event_name: MechEventName
) -> Optional[Union[MechRequest, MechResponse]]:
    """Parse tools content from IPFS."""
    struct = EVENT_TO_MECH_STRUCT.get(event_name)
    raw_content[REQUEST_ID] = str(event.requestId)
    raw_content[BLOCK_FIELD] = str(event.for_block)

    try:
        mech_response = struct(**raw_content)
    except (ValueError, TypeError, KeyError):
        tqdm.write(f"Could not parse {limit_text(str(raw_content))}")
        return None

    if event_name == MechEventName.REQUEST and mech_response.tool in IRRELEVANT_TOOLS:
        return None

    return mech_response


def get_contents(
    session: requests.Session, events: List[MechEvent], event_name: MechEventName
) -> pd.DataFrame:
    """Fetch the tools' responses."""
    contents = []
    for event in tqdm(events, desc=f"Tools' results", unit="results"):
        url = event.ipfs_link(event_name)
        response = request(session, url)
        if response is None:
            tqdm.write(f"Skipping {event=}.")
            continue

        raw_content = parse_ipfs_response(session, url, event, event_name, response)
        if raw_content is None:
            continue

        mech_response = parse_ipfs_tools_content(raw_content, event, event_name)
        if mech_response is None:
            continue
        contents.append(mech_response)
        time.sleep(IPFS_POLL_INTERVAL)

    return pd.DataFrame(contents)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Clean the dataframe."""
    cleaned = df.drop_duplicates()
    cleaned[REQUEST_ID_FIELD] = cleaned[REQUEST_ID_FIELD].astype("str")
    return cleaned


def transform_request(contents: pd.DataFrame) -> pd.DataFrame:
    """Transform the requests dataframe."""
    return clean(contents)


def transform_deliver(contents: pd.DataFrame, full_contents=False) -> pd.DataFrame:
    """Transform the delivers dataframe."""
    unpacked_result = pd.json_normalize(contents.result)
    # # drop result column if it exists
    if "result" in unpacked_result.columns:
        unpacked_result.drop(columns=["result"], inplace=True)

    # drop prompt column if it exists
    if "prompt" in unpacked_result.columns:
        unpacked_result.drop(columns=["prompt"], inplace=True)

    # rename prompt column to prompt_deliver
    unpacked_result.rename(columns={"prompt": "prompt_deliver"}, inplace=True)
    contents = pd.concat((contents, unpacked_result), axis=1)

    if "result" in contents.columns:
        contents.drop(columns=["result"], inplace=True)

    if "prompt" in contents.columns:
        contents.drop(columns=["prompt"], inplace=True)

    return clean(contents)


def gen_event_filename(event_name: MechEventName) -> str:
    """Generate the filename of an event."""
    return f"{event_name.value.lower()}s.csv"


def read_n_last_lines(filename: str, n: int = 1) -> str:
    """Return the `n` last lines' content of a file."""
    num_newlines = 0
    with open(filename, "rb") as f:
        try:
            f.seek(-2, os.SEEK_END)
            while num_newlines < n:
                f.seek(-2, os.SEEK_CUR)
                if f.read(1) == b"\n":
                    num_newlines += 1
        except OSError:
            f.seek(0)
        last_line = f.readline().decode()
    return last_line


def get_earliest_block(event_name: MechEventName) -> int:
    """Get the earliest block number to use when filtering for events."""
    filename = gen_event_filename(event_name)
    if not os.path.exists(filename):
        return 0

    cols = pd.read_csv(filename, index_col=0, nrows=0).columns.tolist()
    last_line_buff = StringIO(read_n_last_lines(filename))
    last_line_series = pd.read_csv(last_line_buff, names=cols)
    block_field = f"{event_name.value.lower()}_{BLOCK_FIELD}"
    return int(last_line_series[block_field].values[0])


def pipeline_step(
    w3: Web3,
    session: requests.Session,
    event_to_contents: Dict[MechEventName, pd.DataFrame],
    event_to_transformer: Dict[MechEventName, Callable],
    mech_to_info: Dict[ChecksumAddress, Tuple[str, int]],
    stop_block: Optional[int] = None,
    start_block: Optional[int] = None,
    full_contents: bool = False,
):
    """Perform a step of the pipeline, from start block (or default earliest) to stop block."""
    for event_name, transformer in event_to_transformer.items():
        events = []
        for address, (abi, earliest_block) in mech_to_info.items():
            if start_block is None:
                start_block = max(earliest_block, get_earliest_block(event_name))
                stop_block = start_block + SNAPSHOT_RATE * BLOCKS_CHUNK_SIZE

            # Parallelize the fetching of events
            with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
                futures = []
                for i in range(start_block, stop_block, BLOCKS_CHUNK_SIZE):
                    futures.append(
                        executor.submit(
                            get_events,
                            w3,
                            event_name.value,
                            address,
                            abi,
                            i,
                            min(i + BLOCKS_CHUNK_SIZE, stop_block),
                        )
                    )

                for future in tqdm(
                    as_completed(futures),
                    total=len(futures),
                    desc=f"Fetching {event_name.value} Events",
                ):
                    current_mech_events = future.result()
                    events.extend(current_mech_events)

        parsed = parse_events(events)

        # Parallelize the fetching of contents; use tqdm to show progress
        contents = []
        with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
            futures = []
            for i in range(0, len(parsed), 1000):
                futures.append(
                    executor.submit(
                        get_contents, session, parsed[i : i + 1000], event_name
                    )
                )

            for future in tqdm(
                as_completed(futures),
                total=len(futures),
                desc=f"Fetching {event_name.value} Contents",
            ):
                current_mech_contents = future.result()
                contents.append(current_mech_contents)

        contents = pd.concat(contents, ignore_index=True)

        if event_name == MechEventName.REQUEST:
            transformed = transformer(contents)
        elif event_name == MechEventName.DELIVER:
            transformed = transformer(contents, full_contents=full_contents)

        events_filename = gen_event_filename(event_name)
        if os.path.exists(events_filename):
            old = pd.read_csv(events_filename)

            # Reset index to avoid index conflicts
            old.reset_index(drop=True, inplace=True)
            transformed.reset_index(drop=True, inplace=True)

            # Concatenate DataFrames
            transformed = pd.concat([old, transformed], ignore_index=True)

            # Drop duplicates if necessary
            transformed.drop_duplicates(inplace=True)

        event_to_contents[event_name] = transformed.copy()

    return stop_block


def store_progress(
    filename: str,
    event_to_contents: Dict[MechEventName, pd.DataFrame],
    tools: pd.DataFrame,
) -> None:
    """Store the given progress."""
    if filename:
        for event_name, content in event_to_contents.items():
            event_filename = gen_event_filename(event_name)
            if "error" in content.columns:
                content.drop(columns=["error"], inplace=True)

            if "result" in content.columns:
                content.drop(columns=["result"], inplace=True)

            content.to_csv(event_filename, index=False, escapechar="\\")

        # drop result and error columns
        if "result" in tools.columns:
            tools.drop(columns=["result"], inplace=True)

        tools.to_csv(filename, index=False, escapechar="\\")


def etl(
    rpcs: List[str], filename: Optional[str] = None, full_contents: bool = True
) -> pd.DataFrame:
    """Fetch from on-chain events, process, store and return the tools' results on all the questions as a Dataframe."""
    w3s = [Web3(HTTPProvider(r)) for r in rpcs]
    session = create_session()
    event_to_transformer = {
        MechEventName.REQUEST: transform_request,
        MechEventName.DELIVER: transform_deliver,
    }
    mech_to_info = {
        to_checksum_address(address): (
            os.path.join(CONTRACTS_PATH, filename),
            earliest_block,
        )
        for address, (filename, earliest_block) in MECH_TO_INFO.items()
    }
    event_to_contents = {}

    latest_block = LATEST_BLOCK
    if latest_block is None:
        latest_block = w3s[0].eth.get_block(LATEST_BLOCK_NAME)[BLOCK_DATA_NUMBER]

    next_start_block = None

    # Loop through events in event_to_transformer
    for event_name, transformer in event_to_transformer.items():
        if next_start_block is None:
            next_start_block_base = get_earliest_block(event_name)

        # Loop through mech addresses in mech_to_info
        events = []
        for address, (abi, earliest_block) in mech_to_info.items():
            if next_start_block_base == 0:
                next_start_block = earliest_block
            else:
                next_start_block = next_start_block_base

            print(
                f"Searching for {event_name.value} events for mech {address} from block {next_start_block} to {latest_block}."
            )

            # parallelize the fetching of events
            with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
                futures = []
                for i in range(
                    next_start_block, latest_block, BLOCKS_CHUNK_SIZE * SNAPSHOT_RATE
                ):
                    futures.append(
                        executor.submit(
                            get_events,
                            random.choice(w3s),
                            event_name.value,
                            address,
                            abi,
                            i,
                            min(i + BLOCKS_CHUNK_SIZE * SNAPSHOT_RATE, latest_block),
                        )
                    )

                for future in tqdm(
                    as_completed(futures),
                    total=len(futures),
                    desc=f"Fetching {event_name.value} Events",
                ):
                    current_mech_events = future.result()
                    events.extend(current_mech_events)

        parsed = parse_events(events)

        contents = []
        with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
            futures = []
            for i in range(0, len(parsed), GET_CONTENTS_BATCH_SIZE):
                futures.append(
                    executor.submit(
                        get_contents,
                        session,
                        parsed[i : i + GET_CONTENTS_BATCH_SIZE],
                        event_name,
                    )
                )

            for future in tqdm(
                as_completed(futures),
                total=len(futures),
                desc=f"Fetching {event_name.value} Contents",
            ):
                current_mech_contents = future.result()
                contents.append(current_mech_contents)

        contents = pd.concat(contents, ignore_index=True)

        full_contents = True
        if event_name == MechEventName.REQUEST:
            transformed = transformer(contents)
        elif event_name == MechEventName.DELIVER:
            transformed = transformer(contents, full_contents=full_contents)

        events_filename = gen_event_filename(event_name)

        if os.path.exists(events_filename):
            old = pd.read_csv(events_filename)

            # Reset index to avoid index conflicts
            old.reset_index(drop=True, inplace=True)
            transformed.reset_index(drop=True, inplace=True)

            # Concatenate DataFrames
            transformed = pd.concat([old, transformed], ignore_index=True)

            # Drop duplicates if necessary
            transformed.drop_duplicates(inplace=True)

        event_to_contents[event_name] = transformed.copy()

    # Store progress
    tools = pd.merge(*event_to_contents.values(), on=REQUEST_ID_FIELD)
    store_progress(filename, event_to_contents, tools)

    return tools


if __name__ == "__main__":
    RPCs = [
        "https://lb.nodies.app/v1/406d8dcc043f4cb3959ed7d6673d311a",
    ]

    tools = etl(rpcs=RPCs, filename=DEFAULT_FILENAME, full_contents=True)
