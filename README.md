# jd-matcher

Ranks candidate profiles against job descriptions with embedding cosine
similarity — no model call per comparison — then deliberately shows
where that approach fails. Embeddings encode what a document is about,
not whether it satisfies a hard constraint; the notebook makes that
argument end-to-end on a small designed corpus.

The whole demo is one notebook: **[`jd_matcher.ipynb`](./jd_matcher.ipynb)**.
It is committed with outputs so it renders on GitHub without anyone
running it.

## Layout

```
src/            small module the notebook imports
  embed.py      OpenRouter client + mock fallback + disk cache
  cost.py       real prices, complexity claim
  filters.py    constraint parsing + structured filter
  profiles.py   load profiles with YAML-frontmatter structured fields
  jds.py        load JDs from data/jds/*.txt
  render.py     heatmap + ranked-list helpers
data/
  jds/          10 job descriptions (2 real Uber, 8 fabricated)
  profiles/     5 candidate profiles (2 real from karrouhq.com, 3 fabricated)
  cache/        embedding + fetched-profile caches (gitignored)
scripts/
  build_notebook.py   rebuilds jd_matcher.ipynb from source cells
```

## Running locally

```
python3 -m pip install -r requirements.txt
python3 scripts/build_notebook.py       # rebuild + execute + save with outputs
jupyter lab jd_matcher.ipynb            # or open in your editor of choice
```

The notebook has a `MOCK = True` toggle at the top. Mock mode uses
deterministic hashed-bag-of-tokens pseudo-embeddings and requires no
network — the fallback path for a live demo behind a bad network. Set
`MOCK = False` to hit OpenRouter (needs `OPENROUTER_API_KEY`; see
`.env.example`). Either way, the cost panel reports real prices from
`src/cost.py`.

## Uber postings

`data/jds/01_uber_swe1_sf.txt` and `02_uber_swe1_nyc.txt` are
placeholders. Paste the posting text into each file; the leading
comment block is stripped by the loader before embedding.

## Honest caveats

Five profiles is a demonstration of method, not evidence of quality. A
real evaluation would need a labelled set — roughly 30 known-good
(profile, JD) pairs annotated by someone with domain judgement — and
would report top-1 accuracy, recall at k, and the rate at which the
structured filter changes the top result. This prototype has none.
