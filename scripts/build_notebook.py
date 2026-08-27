"""Build jd_matcher.ipynb from source cells defined here, then execute it.

The notebook is the artifact users see on GitHub. This script exists so
the source of every cell is version-controllable as Python, not
JSON-diff-hostile ipynb blobs. Re-run whenever you change a cell:

    python scripts/build_notebook.py

Or to only rebuild without executing (faster while iterating):

    python scripts/build_notebook.py --no-exec
"""

from __future__ import annotations

import argparse
from pathlib import Path
from textwrap import dedent

import nbformat
from nbclient import NotebookClient


ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "jd_matcher.ipynb"


def md(text: str):
    return nbformat.v4.new_markdown_cell(dedent(text).strip() + "\n")


def code(text: str):
    return nbformat.v4.new_code_cell(dedent(text).strip() + "\n")


CELLS = [
    md("""
    # jd-matcher

    Ranking candidate profiles against job descriptions with embedding
    cosine similarity — **no model call per comparison** — and then
    deliberately showing where that approach fails.

    The point of this notebook is not the ranking. The point is the
    architectural claim: embedding cosine is the right tool for the
    first pass over a corpus, but it is not a filter over hard
    constraints. Those two jobs are not the same job, and pretending
    they are is where cost gets wasted and wrong answers slip through.

    Structure:

    1. Load the corpus (5 candidate profiles, 10 job descriptions).
    2. Embed everything once. Score every pair by cosine.
    3. Render the full 5x10 matrix and per-JD ranked lists.
    4. Cost panel, framed as a complexity claim.
    5. Break it on purpose: run a query with a hard constraint,
       compare cosine-only to structured-filter-then-cosine.
    6. Draw the pipeline this justifies.
    7. Be honest about what a five-profile demo does and doesn't show.
    """),
    md("""
    ## Setup

    `MOCK=True` uses deterministic pseudo-embeddings so the notebook
    runs end-to-end with no network. The cost panel still reports
    real prices from `src/cost.py` — the arithmetic is the same
    either way; only the vectors change.

    Set `MOCK=False` to hit OpenRouter for real embeddings. That
    requires `OPENROUTER_API_KEY` in the environment (see `.env.example`).
    """),
    code("""
    import os
    import sys
    from pathlib import Path

    import numpy as np
    import pandas as pd
    from dotenv import load_dotenv

    load_dotenv()

    # ============================================================
    # Toggle: True = deterministic pseudo-embeddings, no network.
    #         False = hit OpenRouter for real embeddings.
    # ============================================================
    MOCK = True
    # ============================================================

    sys.path.insert(0, str(Path.cwd()))
    from src import embed, cost, filters, profiles as prof, jds, render

    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not MOCK and not api_key:
        print("No OPENROUTER_API_KEY in env — falling back to MOCK.")
        MOCK = True
    print(f"MOCK = {MOCK}")
    """),
    md("""
    ## 1. The corpus

    Five candidate profiles and ten job descriptions live under
    `data/`. Each profile carries its **structured fields** —
    location, work arrangement, years of experience, seniority — in
    YAML frontmatter, held **separately from the prose**. That
    separation is load-bearing: those fields are the filter layer in
    section 5.

    Two profiles are real (fetched from `karrouhq.com/{user}.md` and
    cached under `data/cache/profiles/`); three are fabricated and
    labelled as such. One of the fabricated profiles is deliberately
    designed to look like an excellent match on prose while violating
    hard constraints — that's the demonstration in section 5.

    Two JDs are placeholders for the two real Uber Software Engineer I
    postings (San Francisco and New York). Paste the posting text into
    `data/jds/01_uber_swe1_sf.txt` and `02_uber_swe1_nyc.txt`; the
    loader strips the leading comment block automatically. The other
    eight JDs are fabricated in the style of real listings and are
    labelled `[Illustrative posting]` at the top.
    """),
    code("""
    profiles_list = prof.load_all()
    jd_list = jds.load_all()

    print(f"Loaded {len(profiles_list)} profiles and {len(jd_list)} JDs.\\n")

    print("Profiles (structured fields shown; prose is what gets embedded):")
    for p in profiles_list:
        tag = " [FABRICATED]" if p.is_fabricated else " [real, fetched]"
        print(f"  {p.display_name}{tag}")
        print(f"    location={p.location!s:<16} arrangement={p.work_arrangement!s:<8} "
              f"years={p.years_of_experience!s:<5} seniority={p.seniority!s}")

    print()
    print("Job descriptions:")
    for j in jd_list:
        print(f"  {j.slug}")
        print(f"    {j.title[:90]}")
    """),
    md("""
    ## 2. Embed once, then it is just linear algebra

    The whole point of using embeddings is that the model cost is paid
    **once per document** at ingest time, and every subsequent
    pairwise comparison is a dot product. Fifteen documents, one round
    of API calls, and then all 50 profile-against-JD scores fall out
    of a single matrix multiply.

    The `UsageLedger` records everything we actually spent: embeddings
    computed (cache misses), tokens sent to the embedding API, and
    chat-model calls (zero — by construction).
    """),
    code("""
    ledger = cost.UsageLedger()

    profile_texts = [p.prose for p in profiles_list]
    jd_texts      = [j.text  for j in jd_list]
    all_texts     = profile_texts + jd_texts

    vecs = embed.embed_many(all_texts, ledger=ledger, mock=MOCK, api_key=api_key)
    prof_vecs = vecs[:len(profile_texts)]
    jd_vecs   = vecs[len(profile_texts):]

    print(f"Embeddings computed this run (cache misses): {ledger.embeddings_computed}")
    print(f"Embeddings served from cache:                {ledger.embeddings_from_cache}")
    print(f"Tokens embedded this run:                    {ledger.tokens_embedded}")
    print(f"Chat-model calls made:                       {ledger.chat_model_calls}")
    print(f"Vector matrix shape (docs x dim):            {vecs.shape}")
    """),
    md("""
    ## 3. The full 5 x 10 matrix

    Every candidate against every posting, no model calls per pair.
    """),
    code("""
    scores = embed.cosine_matrix(prof_vecs, jd_vecs)
    profile_names = [p.display_name for p in profiles_list]
    jd_short = [j.slug for j in jd_list]
    render.heatmap(scores, profile_names, jd_short,
                   title=f"Cosine similarity — {'MOCK' if MOCK else 'real'} embeddings")
    """),
    md("""
    Same data, as a ranked list per JD (best-first). The score is
    printed alongside each name so you can see the spread — in the
    real-embedding run this is where sensible-looking matches
    concentrate, and in the mock run this is basically noise
    (deterministic noise, but noise).
    """),
    code("""
    ranked = render.ranked_list_per_jd(scores, profile_names, jd_short)
    ranked
    """),
    md("""
    ## 4. Cost, stated as a complexity claim

    The naive alternative to this pipeline is: for every (profile, JD)
    pair, prompt a chat model with both documents and ask it to score
    the match. Fifty pairs, fifty calls, and it grows as N x M.

    This pipeline pays the model cost once per document at embedding
    time — 15 embeddings, one round trip — and every subsequent
    comparison is a dot product with a marginal cost of essentially
    nothing.

    Prices used are printed below so the comparison is auditable.
    Update them in `src/cost.py` if OpenRouter changes rates or if you
    want to use a different chat model for the baseline.
    """),
    code("""
    MODEL_EMBED         = "openai/text-embedding-3-small"
    MODEL_CHAT_BASELINE = "openai/gpt-4o"

    # Assumed averages for the naive baseline. Each (profile, JD) call
    # would send both documents plus a scoring prompt wrapper (~900 tok
    # input) and get back a short score and justification (~50 tok).
    AVG_INPUT_PER_PAIR  = 900
    AVG_OUTPUT_PER_PAIR = 50

    embed_usd = ledger.embedding_cost_usd(MODEL_EMBED)
    n_calls, naive_usd = cost.naive_pairwise_cost_usd(
        n_profiles=len(profiles_list),
        n_jds=len(jd_list),
        avg_tokens_per_pair_input=AVG_INPUT_PER_PAIR,
        avg_tokens_per_pair_output=AVG_OUTPUT_PER_PAIR,
        chat_model=MODEL_CHAT_BASELINE,
    )

    print("--- Prices used (edit src/cost.py to update) ---")
    for m, p in cost.PRICES_USD_PER_MTOK.items():
        print(f"  {m:<40}  input ${p['input']:>6.4f}/Mtok   output ${p['output']:>6.4f}/Mtok")

    print()
    print("--- What this pipeline actually spent ---")
    print(f"  embedding model:          {MODEL_EMBED}")
    print(f"  embeddings computed:      {ledger.embeddings_computed}")
    print(f"  tokens embedded:          {ledger.tokens_embedded}")
    print(f"  chat-model calls:         {ledger.chat_model_calls}")
    print(f"  embedding cost:           ${embed_usd:.6f}")

    print()
    print("--- What the naive per-pair approach would have cost ---")
    print(f"  chat model:               {MODEL_CHAT_BASELINE}")
    print(f"  pairwise calls:           {n_calls}")
    print(f"  assumed avg tokens/call:  {AVG_INPUT_PER_PAIR} in + {AVG_OUTPUT_PER_PAIR} out")
    print(f"  naive cost:               ${naive_usd:.4f}")

    print()
    if embed_usd > 0:
        print(f"Ratio naive / embeddings: {naive_usd / embed_usd:,.0f}x")
    else:
        print("Ratio naive / embeddings: effectively infinite (embedding cost rounds to $0)")

    print()
    print(cost.format_complexity_claim(
        n_profiles=len(profiles_list),
        n_jds=len(jd_list),
        k_survivors="k",  # actual k depends on the query; see next section
    ))
    """),
    md("""
    ## 5. Break it on purpose

    This is the point of the exercise.

    A recruiter types a query with a **hard constraint**. Cosine
    similarity, run against the profiles alone, will happily put a
    candidate who violates that constraint at the top — because the
    candidate's prose reads exactly like the query. "5+ years" and
    "3 years" are near-neighbours in embedding space. "Onsite in
    New York" and "remote from Berlin" are, too. The model that made
    the embedding was not asked which side of a predicate a document
    falls on.

    The fix is not a better embedding. The fix is to apply the hard
    constraints as a structured filter **before** cosine ranks the
    survivors. That is the architecture in section 6.

    The query below is a variable — edit it and re-run the cell to
    try other constraints.
    """),
    code("""
    # ============================================================
    # EDIT THIS QUERY AND RE-RUN THE CELL
    # ============================================================
    QUERY = "senior backend engineer, must be onsite in New York, 5+ years Go"
    # ============================================================

    # Parse the query into structured constraints. Deterministic:
    # regex and keyword lookup, no model call. The trace shows what
    # was picked up so nothing is hidden.
    constraints = filters.parse_query(QUERY)
    print("Parsed constraints:")
    for line in constraints.trace:
        print(f"  {line}")
    print()

    # Embed the query itself.
    q_vec = embed.embed_many([QUERY], ledger=ledger, mock=MOCK, api_key=api_key)
    q_scores = embed.cosine_matrix(prof_vecs, q_vec)[:, 0]

    # -------- Ranking A: cosine only --------
    order_a = np.argsort(-q_scores)
    ranking_a = pd.DataFrame([
        {"rank": i + 1,
         "profile": profile_names[j],
         "cosine": round(float(q_scores[j]), 3),
         "location": profiles_list[j].location,
         "arrangement": profiles_list[j].work_arrangement,
         "years": profiles_list[j].years_of_experience}
        for i, j in enumerate(order_a)
    ])

    # -------- Ranking B: filter first, then cosine on survivors --------
    kept, dropped = filters.apply(
        [p.as_filter_dict() for p in profiles_list], constraints
    )
    kept_names = {k["username"] for k in kept}
    surv_idx = [i for i, p in enumerate(profiles_list) if p.username in kept_names]

    if surv_idx:
        surv_scores = q_scores[surv_idx]
        order_b = np.argsort(-surv_scores)
        ranking_b = pd.DataFrame([
            {"rank": i + 1,
             "profile": profile_names[surv_idx[j]],
             "cosine": round(float(q_scores[surv_idx[j]]), 3),
             "location": profiles_list[surv_idx[j]].location,
             "arrangement": profiles_list[surv_idx[j]].work_arrangement,
             "years": profiles_list[surv_idx[j]].years_of_experience}
            for i, j in enumerate(order_b)
        ])
    else:
        ranking_b = pd.DataFrame(columns=["rank", "profile", "cosine",
                                          "location", "arrangement", "years"])

    print("=" * 78)
    print(f"Query: {QUERY!r}")
    print("=" * 78)
    print()
    print("RANKING A  —  cosine only (no structured filter)")
    print("-" * 78)
    print(ranking_a.to_string(index=False))
    print()
    print("RANKING B  —  structured filter first, then cosine on survivors")
    print("-" * 78)
    if ranking_b.empty:
        print("(no profile in the set satisfies all hard constraints)")
    else:
        print(ranking_b.to_string(index=False))
    print()

    if dropped:
        print("Rejected by the structured filter (with reasons):")
        for d in dropped:
            print(f"  {d['username']}: " + "; ".join(d["_reject_reasons"]))
    """),
    md("""
    Read the two rankings against each other. Any profile that appears
    high in A and is missing from B was pruned by a hard constraint
    — the rejection reasons are printed under the table.

    The pattern to notice:

    - **Cosine encodes what a document is about**, not whether it
      satisfies a predicate. A profile that talks fluently about
      senior backend Go work will score high against a query that
      asks for senior backend Go work — whether or not the candidate
      is where you need them and has the years you need.
    - **"5+ years" and "3 years" are near-neighbours in embedding
      space.** They read almost identically. The difference between
      them is a fact, not a topic. Facts belong in a filter.
    - **The filter is not competing with the ranker.** Cosine still
      does the semantic work; the filter just prevents the wrong kind
      of top-1.
    """),
    md("""
    ## 6. The architecture this justifies

    ```
                +-------------------------+
                |  ingest / update time   |
                +-------------------------+
                            |
                            v
                +-------------------------+       one-time cost
                |     EMBED once          |       ~$0.02 / 1M tokens
                |  (per document)         |       cached to disk
                +-------------------------+
                            |
                            v
    query time:
                +-------------------------+
                |  EMBED query            |       one small call
                +-------------------------+
                            |
                            v
                +-------------------------+       O(N) cheap
                |  cosine over corpus     |       dot products
                |  -> top-M shortlist     |       no model calls
                +-------------------------+
                            |
                            v
                +-------------------------+       O(M) rule application
                |  STRUCTURED FILTER      |       deterministic
                |  (hard constraints)     |       explainable
                |  -> k survivors         |
                +-------------------------+
                            |
                            v
                +-------------------------+       O(k) model calls
                |  chat model on          |       full reasoning only
                |  survivors only         |       where it can change
                |                         |       the answer
                +-------------------------+
    ```

    The claim is not that cosine is a substitute for a model call. The
    claim is that cosine and structured filtering are the right way to
    **spend** the model-call budget — on the small set of candidates
    where careful reasoning can still change the outcome, not on the
    entire corpus.

    - Naive: **O(N) model calls** across the whole corpus.
    - This pipeline: **O(N) cheap vector ops + O(k) model calls on
      survivors**, where k is typically a small integer.

    For 5 profiles x 10 JDs, the difference is measured in cents. At
    5,000 profiles x 500 JDs the difference is what makes the product
    possible at all.
    """),
    md("""
    ## 7. Honest note on what this demonstrates

    Five profiles is a demonstration of **method**, not evidence of
    **quality**. Nothing here shows that this ranker would be right on
    a real hiring pipeline, only that the mechanism works as claimed
    on a small designed set.

    A real evaluation would need a labelled set — roughly 30
    known-good (profile, JD) pairs annotated by someone with domain
    judgement — and would report top-1 accuracy, recall at k, and the
    rate at which the structured filter changes the top result. This
    prototype has none of that.

    It also has two other honest caveats worth naming:

    - **The structured fields are hand-annotated.** The YAML
      frontmatter in each profile was written by a human looking at
      the prose. In production, that annotation is itself a modelling
      problem — extractable, but not free.
    - **The filter vocabulary is small on purpose.** `src/filters.py`
      recognises a fixed set of cities, arrangements, and skills.
      Extending it is trivial and unglamorous; pretending it's clever
      would be dishonest.

    Take this notebook as a shape-of-the-argument artifact, not a
    benchmark.
    """),
]


def build() -> nbformat.NotebookNode:
    nb = nbformat.v4.new_notebook()
    nb.cells = CELLS
    nb.metadata = {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python", "pygments_lexer": "ipython3"},
    }
    return nb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-exec", action="store_true",
                    help="Write the notebook without executing it.")
    args = ap.parse_args()

    nb = build()

    if not args.no_exec:
        print("Executing notebook (this runs the pipeline end-to-end)...")
        client = NotebookClient(nb, timeout=120, kernel_name="python3",
                                resources={"metadata": {"path": str(ROOT)}})
        client.execute()

    nbformat.write(nb, OUT)
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
