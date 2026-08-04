# CVLora

This repo has **two separate Streamlit apps**:

| File | What it is | Data |
|---|---|---|
| `candidate_matcher_app.py` | The **real** backend app — loads the fine-tuned LoRA model (`Qwen2.5-0.5B-Instruct` + `safaa99/cvlora`), extracts structured candidate data from real resumes (PDF/DOCX/TXT), matches candidates against chosen skills, and calculates ROI from a real `evaluation_report.json`. | Real (JSON files in `data/outputs`, real model inference) |
| `app.py` + `pages/` | The **premium UI/UX prototype** — same visual design goal (Deep Navy / Royal Blue / Emerald, Inter + Plus Jakarta Sans), but with mock data so every screen (Dashboard, Resume Analysis, Candidate Search, Ranking, Company Dashboard, ROI Analytics, Settings) can be previewed without running the real model. | Mock (`utils/sample_data.py`) |

They are **not wired together yet** — see "Connecting the two" below.

## Run the real backend app
```bash
pip install -r requirements.txt torch transformers peft pypdf docx2txt
streamlit run candidate_matcher_app.py
```

## Run the UI/UX prototype

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Pages
- **app.py** — Dashboard overview (KPIs, trend, funnel, top candidates, activity)
- **pages/1 Resume Analysis** — Upload + AI extraction with confidence indicators
- **pages/2 Candidate Search** — Filterable candidate search with premium cards
- **pages/3 Candidate Ranking** — Sortable ranking table + candidate detail drill-down
- **pages/4 Company Dashboard** — Executive view: funnel, distributions, skill heatmap
- **pages/5 ROI Analytics** — Full & hybrid ROI model, wired to accept a real
  `evaluation_report.json` from your existing `candidate_matcher_app.py` pipeline
- **pages/6 Settings** — Workspace, model, and notification settings

## Design system
Shared tokens, CSS, and reusable components (KPI cards, match badges,
progress bars, skill tags) live in `utils/style.py`. Mock data lives in
`utils/sample_data.py` — swap this out for your real extraction/ranking
pipeline output.

## Connecting the two

`candidate_matcher_app.py` is a standalone Streamlit script (it calls
`st.set_page_config()` itself and runs top-to-bottom), so it can't be
imported directly as-is. To wire the real backend into the new UI:

1. Split the reusable logic out of `candidate_matcher_app.py` into plain
   functions in a new file, e.g. `core/engine.py`:
   - `load_model(...)`
   - `run_resume_extraction(...)`
   - `load_candidates_from_json(directory)`
   - the ROI math (already almost identical to `pages/5_💰_ROI_Analytics.py`)
2. In `utils/sample_data.py`, replace `generate_candidates()` with a call to
   `load_candidates_from_json("data/outputs")` so `Candidate Search` /
   `Candidate Ranking` / `Company Dashboard` show real extracted candidates.
3. In `pages/1_📄_Resume_Analysis.py`, replace the simulated `time.sleep()`
   progress loop with a real call to `run_resume_extraction()`.
4. In `pages/5_💰_ROI_Analytics.py`, the `evaluation_report.json` loader is
   already there — just point `eval_path` at the real file produced by
   `full_evaluation.py`.

Until that split is done, run `candidate_matcher_app.py` for real
extraction/matching, and `app.py` to preview/iterate on the design.
