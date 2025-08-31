# PapeRank for Literature Surveys: Practical Use Cases

PapeRank helps you quickly find and prioritize the most relevant papers to read by looking at how articles cite each other around your topic of interest. It expands from papers you know (or a short list you trust), builds a small citation network, and ranks papers by how central they are within that network.

Notes
- Works with a local “neighborhood” around your starting papers (not the entire literature).
- Uses public data sources (Crossref and OpenCitations); some references may be missing DOIs.
- Exports ranked results with authors, title, year to JSON/CSV.
- Exports default to the top 10 items; override with max_results in rank_and_save_publications_JSON/CSV.
- For steps>1, crawling expands via iterative 1‑hop unions with deduplication across steps (each DOI is visited at most once).

Prerequisites
- Optional: install tqdm to see progress bars (pip install tqdm). Set progress='tqdm' or True.
- Recommended: set an email for polite Crossref usage:
  - macOS/Linux terminal: export CROSSREF_MAILTO="you@example.com"
- Networked functions use caching internally; results still depend on API availability.

---

## Use Case 1 — “I have one cornerstone paper. What else should I read next?”

Scenario
- You’re drafting a survey on a topic. You have one well-known paper and want a focused, defensible reading list that covers key predecessors (papers it cites) and successors (papers that cite it).

What you do
1) Start from your cornerstone DOI.
2) Ask PapeRank to gather its nearby network (who it cites and who cites it) and rank papers by importance within that local network.
3) Skim the top-ranked results first and use the export to justify your selection.

Minimal example (Python)
```python
from paperank.paperank_core import crawl_and_rank_frontier

results = crawl_and_rank_frontier(
    doi="10.1016/j.ejor.2016.12.001",  # your cornerstone DOI
    steps=1,                           # 1-hop bidirectional frontier (cited + citing)
    alpha=0.85,                        # sensible default
    output_format="csv",               # also supports "json"
    debug=False,
    progress=True
)

# results is a list of (doi, score), highest first
for doi, score in results[:10]:
    print(f"{score:.5f}  {doi}")
# A CSV file like 10_1016_j_ejor_2016_12_001.csv is saved with authors, title, year.
```
Notes
- steps=1 collects both cited and citing 1-hop neighbors around each seed (bidirectional frontier).
- For steps>1, the crawl performs iterative 1‑hop unions with deduplication across steps (frontier-style; nodes are not revisited).
- Multiple seeds are supported by passing a list to `doi`, e.g., `doi=["doiA", "doiB"]` (output file base name becomes `crawl_{num_seeds}_seeds`).
- You can filter during crawling with `min_year` and `min_citations`:
```python
results = crawl_and_rank_frontier(
    doi="10.1016/j.ejor.2016.12.001",
    steps=1,
    min_year=2000,
    min_citations=5,
    progress=True
)
```

How this helps your survey
- Build a reading plan: Read the top 10–20 first; they are central within the immediate topic network.
- Spot context: Backward links (papers cited by the cornerstone) expose foundations. Forward links (papers that cite it) show follow-ups and applications.
- Justify inclusion: “We prioritized the top N items by PapeRank (a citation-network centrality) within the local neighborhood of our cornerstone paper.”

Interpretation tips
- Central ≠ most cited globally. It means “well-connected within this focused topic.”
- If results look too narrow, increase steps (e.g., steps=2). If too broad, reduce steps.
- Keep a short note of parameters (steps, α) and the date to make your selection reproducible.

---

## Use Case 2 — “I have a shortlist of 20–50 DOIs; which ones truly anchor the field?”

Scenario
- You’ve gathered 20–50 DOIs from keyword searches, a review article, or colleagues’ suggestions. You want to quickly identify the core set to read first and to cite with confidence.

What you do
1) Put your DOIs into a simple list (from your notes, a spreadsheet, or a text file).
2) Ask PapeRank to rank only within this list, surfacing the most central items.
3) Skim the top-ranked papers first; optionally export the top N with full metadata for your notes.

Minimal example (Python)
```python
from paperank.paperank_core import rank, rank_and_save_publications_CSV

doi_list = [
    "10.1016/j.ejor.2016.12.001",
    "10.1080/1540496x.2019.1696189",
    "10.1016/j.intfin.2017.09.008",
    "10.1287/opre.2020.1971",
    "10.1287/mnsc.2019.3375",
    "10.1137/16M1058204",
    # ...add the rest of your shortlist...
]

ranked = rank(doi_list, alpha=0.85, debug=False, progress=True)
for doi, score in ranked[:10]:
    print(f"{score:.5f}  {doi}")

# Optional: save a compact reading pack (top 15) with authors/title/year
rank_and_save_publications_CSV(doi_list, out_path="shortlist_top15.csv", alpha=0.85, max_results=15)
```

How this helps your survey
- Triage reading order: Start with the top 10–20 central papers to build a solid backbone.
- Filter noise: Items that are peripheral within your shortlist fall down the ranking—review them later or drop if off-topic.
- Defensible selection: “From our candidate list, we prioritized the top N papers by PapeRank centrality within the shortlist.”

Interpretation tips
- Keep the list coherent: Remove obviously off-topic items and rerun; scores reflect centrality within your list.
- Split if needed: If your shortlist mixes subtopics, run each subgroup separately to find anchors per subtopic.
- Check connectivity: If many items score similarly low, your list may be too broad—refine the shortlist or add a few known connectors.

---

## Use Case 3 — “I need ‘bridging’ works that connect two subtopics.”

Scenario
- Your survey spans two related subtopics (e.g., robust optimization and stochastic programming). You want the few papers that connect both strands so you can explain how ideas flowed across them.

What you do
1) Pick one representative paper (DOI) for each subtopic.
2) Ask PapeRank to gather a small neighborhood around each seed (who they cite and who cites them).
3) Look at the overlap between the two neighborhoods—these are natural “bridges.”
4) Rank the union of both neighborhoods to surface additional central connectors you may have missed.

Minimal example (Python)
```python
from paperank.citation_crawler import get_citation_neighborhood
from paperank.paperank_core import rank

seed_a = "10.1016/j.ejor.2016.12.001"  # subtopic A
seed_b = "10.1080/1540496x.2019.1696189"  # subtopic B

nh_a = set(get_citation_neighborhood(seed_a, forward_steps=1, backward_steps=1, progress=True))
nh_b = set(get_citation_neighborhood(seed_b, forward_steps=1, backward_steps=1, progress=True))

overlap = sorted(nh_a & nh_b)          # candidate bridges (appear near both seeds)
shortlist = sorted(nh_a | nh_b)        # union to rank more broadly

ranked = rank(shortlist, alpha=0.85, debug=False, progress=True)

print("Bridging candidates (overlap):")
for d in overlap[:10]:
    print("-", d)

print("\nTop-ranked within the union:")
for d, s in ranked[:10]:
    print(f"{s:.5f}  {d}")
```

How this helps your survey
- Tell a coherent story: Use bridging papers to explain how concepts, methods, or datasets moved between subtopics.
- Choose exemplars: Highlight 2–5 bridges as anchors for your “connections” section.
- Avoid blind spots: Ranking the union often elevates strong connectors even if they don’t appear in the literal overlap.

Interpretation tips
- If overlap is tiny, increase steps (e.g., 2) or try a different seed per subtopic.
- If results are too broad, reduce steps to keep the focus tight.
- Skim abstracts of overlap + top-ranked items first; confirm the cross-link in their references and citations.

---

## Use Case 4 — “I must justify which papers made the cut in the methods section.”

Scenario
- You need a clear, defensible rationale for which papers were prioritized in your review. Supervisors, reviewers, or editors expect transparent criteria and a brief, reproducible description.

What you do
1) Export a ranked list (JSON or CSV) with authors/title/year so you can cite your selection criteria and include a table if needed.
2) Add a short, plain-language methods paragraph to your manuscript.
3) Keep a note of parameters (steps, α) and the retrieval date.

Option A — From one cornerstone DOI (top 10 saved automatically)
```python
from paperank.paperank_core import crawl_and_rank_frontier

# Writes a JSON file like 10_1016_j_ejor_2016_12_001.json with top results
_ = crawl_and_rank_frontier(
    doi="10.1016/j.ejor.2016.12.001",
    steps=1,            # 1-hop bidirectional frontier
    alpha=0.85,
    output_format="json",
    debug=False,
    progress=True
)
# Tip: You can also pass multiple seeds:
# _ = crawl_and_rank_frontier(doi=["doiA", "doiB"], steps=1, ...)
```

Option B — From a shortlist you already have (choose how many to save)
```python
from paperank.paperank_core import rank_and_save_publications_JSON

doi_list = [
    "10.1016/j.ejor.2016.12.001",
    "10.1080/1540496x.2019.1696189",
    "10.1016/j.intfin.2017.09.008",
    # ...your list...
]

# Save a methods-ready JSON file with the top 20 and metadata
rank_and_save_publications_JSON(
    doi_list,
    out_path="methods_top20.json",
    alpha=0.85,
    max_results=20
)
```

Optional: Read the JSON to prepare a quick appendix/table
```python
import json

with open("methods_top20.json", "r", encoding="utf-8") as f:
    payload = json.load(f)

# items contain: rank, doi, score, authors, title, year
for item in payload["items"]:
    print(f'{item["rank"]:>2}. {item["year"]} — {item["title"]} ({item["doi"]})  score={item["score"]:.4f}')
```

Template text you can adapt (methods section)
- “We prioritized publications using PapeRank, a citation-network centrality computed on a local neighborhood around our starting set. For the cornerstone workflow, we expanded one step to predecessors (cited) and successors (citing) and ranked items with damping α = 0.85. For the shortlist workflow, we ranked only within our curated list. We report the top N items (authors, title, year, DOI) and include scores in the appendix. Data were retrieved from Crossref and OpenCitations on [DATE].”

How this helps your survey
- Transparent selection: A concise, defensible criterion for inclusion order.
- Easy appendix: Exported JSON/CSV can be dropped into a table of top N items.
- Reproducible: Record α, steps, and retrieval date to allow reruns.

---

## Use Case 5 — “Have I missed any central papers?”

Scenario
- You already have a working bibliography. Before submitting a survey or proposal, you want a quick sanity check to ensure no central papers in your topic’s neighborhood were overlooked.

What you do
1) Take your current bibliography (the DOIs you plan to cite).
2) Add a small neighborhood around 1–2 cornerstone DOIs.
3) Rank the union, then flag top-ranked items that aren’t yet in your bibliography.

Minimal example (Python)
```python
from paperank.citation_crawler import get_citation_neighborhood
from paperank.paperank_core import rank

# Your current bibliography (sample)
my_bib = [
    "10.1016/j.ejor.2016.12.001",
    "10.1080/1540496x.2019.1696189",
    # ...your current DOIs...
]

# Expand around a cornerstone (you can add a second seed and merge)
seed = "10.1016/j.ejor.2016.12.001"
nh = get_citation_neighborhood(seed, forward_steps=1, backward_steps=1, progress=True)

# Union of what you have and what’s nearby
shortlist = list(dict.fromkeys(my_bib + nh))

# Rank within this union
ranked = rank(shortlist, alpha=0.85, debug=False, progress=True)

# Gaps: top-ranked items not already in your bibliography
bib_set = set(my_bib)
gaps = [(d, s) for d, s in ranked[:20] if d not in bib_set]

print("Potential gaps to review (top-ranked not yet in your bib):")
for d, s in gaps:
    print(f"{s:.5f}  {d}")
```

How this helps your survey
- Reduce blind spots: Quickly surface central items you didn’t plan to cite.
- Prioritize fixes: Review the few “gaps” first; integrate those that are clearly on-topic.
- Document due diligence: Mention that you validated coverage by ranking a local neighborhood.

Interpretation tips
- If many gaps are off-topic, tighten the seeds or reduce steps.
- If important items appear just below the top N, skim a slightly larger cut (e.g., top 20–30).
- Record your parameters (steps, α) and run date for transparency.

---

## Conclusions

You now have five practical workflows: cornerstone expansion, shortlist triage, finding bridges, documenting selection, and coverage sanity-checks. Mix and match them to build a focused, well-justified, and defensible survey.
