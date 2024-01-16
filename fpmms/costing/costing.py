import tiktoken
import anthropic
from typing import Optional
from mech.tools.prediction_request import prediction_request
from mech.tools.prediction_request_sme import prediction_request_sme
from mech.tools.prediction_request_claude import prediction_request_claude


class TokenCounter:
    """
    A class to handle token encoding and counting for different GPT models.
    """

    @staticmethod
    def encoding_for_model(model: str):
        return tiktoken.encoding_for_model(model)

    @staticmethod
    def count_tokens(text: str, model: str) -> int:
        if "claude" in model:
            return anthropic.Anthropic().count_tokens(text)

        enc = TokenCounter.encoding_for_model(model)
        return len(enc.encode(text))


class TokenCostCalculator:
    """
    A class to calculate the cost of tokens for various GPT models and token types.
    """

    TOKEN_PRICES = {
        "gpt-3.5-turbo": {"input": 0.001, "output": 0.002},
        "gpt-4": {"input": 0.03, "output": 0.06},
        "gpt-4-turbo": {"input": 0.01, "output": 0.03},
        "claude-2": {"input": 0.008, "output": 0.024},
    }

    @staticmethod
    def token_to_cost(tokens: int, model: str, tokens_type: str) -> float:
        if model in TokenCostCalculator.TOKEN_PRICES and tokens_type in [
            "input",
            "output",
        ]:
            price_per_thousand = TokenCostCalculator.TOKEN_PRICES[model][tokens_type]
            return tokens / 1000 * price_per_thousand
        else:
            raise ValueError("Unknown model or token type")


class CostCounter:
    """
    A class to calculate the cost of using various tools including token costs and API call costs.
    """

    TOOL_CONFIG = {
        "prediction-offline": {
            "model": "gpt-3.5-turbo",
            "search": 0,
            "final_base_prompt": prediction_request.PREDICTION_PROMPT,
            "search_query_prompt": False,
        },
        "prediction-online": {
            "model": "gpt-3.5-turbo",
            "search": 5,
            "final_base_prompt": prediction_request.PREDICTION_PROMPT,
            "search_query_prompt": prediction_request.URL_QUERY_PROMPT,
        },
        "prediction-online-summarized-info": {
            "model": "gpt-3.5-turbo",
            "search": 5,
            "final_base_prompt": prediction_request.PREDICTION_PROMPT,
            "search_query_prompt": prediction_request.URL_QUERY_PROMPT,
        },
        "prediction-offline-sme": {
            "model": "gpt-3.5-turbo",
            "search": 0,
            "final_base_prompt": prediction_request_sme.PREDICTION_PROMPT,
            "sme_prompt": prediction_request_sme.SME_GENERATION_SYSTEM_PROMPT,
            "search_query_prompt": False,
        },
        "prediction-online-sme": {
            "model": "gpt-3.5-turbo",
            "search": 5,
            "final_base_prompt": prediction_request_sme.PREDICTION_PROMPT,
            "sme_prompt": prediction_request_sme.SME_GENERATION_SYSTEM_PROMPT,
            "search_query_prompt": prediction_request.URL_QUERY_PROMPT,
        },
        "claude-prediction-online": {
            "model": "claude-2",
            "search": 5,
            "final_base_prompt": prediction_request_claude.PREDICTION_PROMPT,
            "search_query_prompt": prediction_request.URL_QUERY_PROMPT,
        },
        "claude-prediction-offline": {
            "model": "claude-2",
            "search": 0,
            "final_base_prompt": prediction_request_claude.PREDICTION_PROMPT,
            "search_query_prompt": False,
        },
    }
    COST_PER_API_CALL = 0.005  # $5 per 1000 API calls
    DEFAULT_OUTPUT_TOKENS = 100

    def __init__(
        self, tool: str, prediciton_prompt: str, final_response: Optional[str] = None
    ):
        # Input variables
        self.tool = tool
        self.prediciton_prompt = prediciton_prompt  # assumes additional information is already added to the prompt
        self.prediciton_response = (
            final_response  # usually this is None; we use 100 tokens as default
        )

        # Cost variables
        self.prediciton_response_cost = 0.0
        self.google_search_cost = 0.0
        self.search_query_cost = 0.0
        self.sme_prompt_cost = 0.0
        self.final_cost = 0.0

        # Token count variables
        self.total_input_tokens = 0
        self.total_output_tokens = 0

    def get_cost(
        self, input: str, output: Optional[str] = None, model: str = "gpt-3.5-turbo"
    ) -> float:
        input_tokens = TokenCounter.count_tokens(input, model)
        self.total_input_tokens += input_tokens
        input_cost = TokenCostCalculator.token_to_cost(input_tokens, model, "input")

        if output:
            output_tokens = TokenCounter.count_tokens(output, model)
            self.total_output_tokens += output_tokens
            output_cost = TokenCostCalculator.token_to_cost(
                output_tokens, model, "output"
            )
        else:
            self.total_output_tokens += self.DEFAULT_OUTPUT_TOKENS
            output_cost = TokenCostCalculator.token_to_cost(
                self.DEFAULT_OUTPUT_TOKENS, model, "output"
            )  # 200 tokens if response is not provided

        return input_cost + output_cost

    def run(self) -> float:
        config = self.TOOL_CONFIG.get(self.tool, {})

        if "offline" in self.tool:
            if self.tool in ["prediction-offline", "claude-prediction-offline"]:
                # Offline tool cost
                final_prompt = self.prediciton_prompt
                self.prediciton_response_cost = self.get_cost(
                    final_prompt, self.prediciton_response, config["model"]
                )

            if self.tool == "prediction-offline-sme":
                # Offline tool cost
                final_prompt = self.prediciton_prompt
                self.prediciton_response_cost = self.get_cost(
                    final_prompt, self.prediciton_response, config["model"]
                )

                # sme prompt cost
                sme_prompt = config["sme_prompt"] + self.prediciton_prompt
                self.sme_prompt_cost = self.get_cost(sme_prompt, None, config["model"])

        else:
            if self.tool == "prediction-online-sme":
                # sme prompt cost
                sme_prompt = config["sme_prompt"] + self.prediciton_prompt
                self.sme_prompt_cost = self.get_cost(sme_prompt, None, config["model"])

            # Online tool cost
            self.google_search_cost = config["search"] * self.COST_PER_API_CALL

            # LLM call to generate search queries cost
            search_query_prompt = config["search_query_prompt"] + self.prediciton_prompt
            self.search_query_cost = self.get_cost(
                search_query_prompt, None, config["model"]
            )

            # Final LLM call cost including additional information from Google search
            final_prompt = self.prediciton_prompt
            self.prediciton_response_cost = self.get_cost(
                final_prompt, self.prediciton_response, config["model"]
            )

        self.final_cost = (
            self.prediciton_response_cost
            + self.google_search_cost
            + self.search_query_cost
            + self.sme_prompt_cost
        )

        return self.final_cost


if __name__ == "__main__":
    # Example usage
    tool = "prediction-offline-sme"
    prediciton_prompt = "The following is a conversation with an AI assistant. The assistant is helpful, creative, clever, and very friendly. Human: Hello, who are you?"
    final_response = "I am an AI created by OpenAI. How can I help you today?"
    cost_counter = CostCounter(tool, prediciton_prompt, final_response)
    cost_counter.run()
    print(cost_counter.final_cost)