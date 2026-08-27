"""Price table and cost math.

Prices are stored explicitly so the notebook can print them alongside
the numbers, making the comparison auditable rather than asserted.
Update these if OpenRouter or the underlying provider changes prices.
"""

from dataclasses import dataclass


# USD per 1,000,000 tokens. These are the prices the notebook will
# print on screen when it reports the cost of the pipeline. If they
# drift from what OpenRouter actually charges, update them here.
PRICES_USD_PER_MTOK = {
    "openai/text-embedding-3-small": {"input": 0.02, "output": 0.0},
    # Chat model used only for the naive baseline comparison. Whatever
    # model we would have called once per (profile, JD) pair.
    "openai/gpt-4o":                  {"input": 2.50, "output": 10.00},
}


@dataclass
class UsageLedger:
    """What we actually spent to produce the ranking."""
    embeddings_computed: int = 0        # cache misses only
    embeddings_from_cache: int = 0
    tokens_embedded: int = 0            # tokens sent to the embedding API
    chat_model_calls: int = 0           # should be zero in this pipeline

    def embedding_cost_usd(self, model: str) -> float:
        return (self.tokens_embedded / 1_000_000.0) * PRICES_USD_PER_MTOK[model]["input"]


def naive_pairwise_cost_usd(
    n_profiles: int,
    n_jds: int,
    avg_tokens_per_pair_input: int,
    avg_tokens_per_pair_output: int,
    chat_model: str,
) -> tuple[int, float]:
    """The cost we DIDN'T pay by using embeddings.

    Naive = one chat-model call per (profile, JD) pair to score the match.
    Returns (n_calls, usd_cost).
    """
    n_calls = n_profiles * n_jds
    prices = PRICES_USD_PER_MTOK[chat_model]
    input_cost = (n_calls * avg_tokens_per_pair_input / 1_000_000.0) * prices["input"]
    output_cost = (n_calls * avg_tokens_per_pair_output / 1_000_000.0) * prices["output"]
    return n_calls, input_cost + output_cost


def format_complexity_claim(
    n_profiles: int,
    n_jds: int,
    k_survivors: int,
) -> str:
    """Frame the win as a complexity claim, not a single number."""
    n = n_profiles * n_jds
    return (
        f"Naive:     O(N) model calls where N = {n_profiles} profiles x "
        f"{n_jds} JDs = {n}.\n"
        f"This one:  O(N) cheap vector ops + O(k) model calls on "
        f"survivors, where k = {k_survivors} after structured filtering.\n"
        f"Both stages are needed. The vector stage is not a substitute "
        f"for reasoning; it is a way to spend the reasoning budget only "
        f"where it can change the answer."
    )
