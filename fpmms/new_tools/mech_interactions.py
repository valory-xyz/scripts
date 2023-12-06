#   ------------------------------------------------------------------------------
#
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

"""Package for interacting with the mech"""

from multiprocessing import Process, Manager
from typing import Dict

from mech_client.interact import interact

AGENT_ID = 6
PRIVATE_KEY_PATH = "./gnosis_exp_pkey.txt"
DEFAULT_TIMEOUT = 600


def call_tool(
    tool_name: str, tool_input: str, timeout: int = DEFAULT_TIMEOUT
) -> Dict[str, str]:
    """Call a mech's tool with a timeout."""
    manager = Manager()
    result = manager.dict()
    p = Process(target=call_tool_worker, args=(tool_name, tool_input, result))
    p.start()

    # wait for timeout seconds or until the process finishes
    p.join(timeout)
    if p.is_alive():
        p.terminate()
        p.join()
        result.update({"error": f"Mech call timed out after {timeout} seconds."})

    return result


def call_tool_worker(tool: str, prompt: str, result: Dict[str, str]) -> None:
    """A worker to call a mech's tool and update the given result."""
    try:
        result.update(interact(prompt, AGENT_ID, tool, PRIVATE_KEY_PATH))
    except Exception as exc:
        result.update(
            {
                "error": f"There was an error while trying to send {prompt=} to {tool=}: {exc}"
            }
        )
