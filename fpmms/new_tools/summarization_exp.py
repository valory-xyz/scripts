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

"""Experimentation with the tools that summarize online information, using the mech client."""

import os.path
from dataclasses import dataclass
from typing import Optional, Any, Iterator, Tuple

import pandas as pd
from tqdm import tqdm

from fpmms.new_tools.mech_interactions import call_tool
from fpmms.tools import MechResponse as MechResponseBase

FPMMS_PATH = "../fpmms.csv"
RESULTS_PATH = "summarization_experiment.csv"
SIMPLE_TOOL = "prediction-online-summarized-info"
POWERFUL_TOOL = "prediction-online-sum-url-content"
N_SAMPLES = 100
BATCH_SIZE = 1
SEED = 0
PROMPT_TEMPLATE = (
    "Please take over the role of a Data Scientist to evaluate the given question. "
    'With the given question "{question}" and the `yes` option represented by `Yes` '
    "and the `no` option represented by `No`, what are the respective probabilities of `p_yes` and `p_no` occurring?"
)


@dataclass
class MechResponse(MechResponseBase):

    title: str

    def __init__(self, **kwargs: Any) -> None:
        """Initialize the mech's response ignoring extra keys."""
        super().__init__(**kwargs)
        self.title = kwargs.pop("title", "")


def load_unique_questions(prev_results: Optional[pd.DataFrame], n_samples: int) -> pd.Series:
    """Load the FPMM data for the experiment, sample a random subset with unique questions, and return them."""
    fpmms = pd.read_csv(FPMMS_PATH)
    titles: pd.Series = fpmms["title"].drop_duplicates()

    if prev_results is not None:
        no_errors_mask = prev_results["error"] == ""
        prev_titles_without_errors = prev_results.loc[no_errors_mask, "title"].str
        titles = titles.loc[titles != prev_titles_without_errors]

    n_samples = min(n_samples, len(titles.index))
    return titles.sample(n_samples, random_state=SEED)


def load_existing() -> Tuple[Optional[pd.DataFrame], int]:
    """Load existing results from previous runs."""
    if os.path.isfile(RESULTS_PATH):
        prev_results = pd.read_csv(RESULTS_PATH)
        return prev_results, len(prev_results.index)
    else:
        return None, 0


def get_raw_responses(questions: pd.Series) -> Iterator[MechResponse]:
    """Get the raw responses."""
    for question in tqdm(questions):
        prompt = PROMPT_TEMPLATE.format(question=question)
        for tool in (SIMPLE_TOOL, POWERFUL_TOOL):
            raw_response = call_tool(tool, prompt)
            raw_response["title"] = question

            try:
                mech_response = MechResponse(**raw_response)
            except (ValueError, TypeError, KeyError):
                mech_response = MechResponse(
                    error=f"Could not parse {str(raw_response)}"
                )

            yield mech_response


def update_data(prev_results: pd.DataFrame, experiment_data: pd.DataFrame) -> pd.DataFrame:
    """Update the experiment data."""
    if prev_results is not None:
        prev_results.dropna(subset="result", inplace=True)
        experiment_data = pd.concat((prev_results, experiment_data))
    return experiment_data


def run_batch(prev_results: Optional[pd.DataFrame], n_samples_so_far: int) -> int:
    """Run a batch of the experiment."""
    n_samples_remaining = N_SAMPLES - n_samples_so_far
    batch_size = min(BATCH_SIZE, n_samples_remaining)
    questions = load_unique_questions(prev_results, batch_size)
    responses = [response for response in get_raw_responses(questions)]
    experiment_data = pd.DataFrame(responses)
    unpacked_result = pd.json_normalize(experiment_data.result)
    experiment_data = pd.concat((experiment_data, unpacked_result), axis=1)
    experiment_data = update_data(prev_results, experiment_data)
    experiment_data.to_csv(RESULTS_PATH, index=False)

    return len(experiment_data.index)


def run_experiment() -> None:
    """Run the summarization tools' experiment."""
    prev_results, n_samples_so_far = load_existing()
    while n_samples_so_far < N_SAMPLES:
        n_samples_new = run_batch(prev_results, n_samples_so_far)
        n_samples_so_far += n_samples_new


if __name__ == "__main__":
    run_experiment()
