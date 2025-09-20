# What to Improve (Precise, Actionable)

## 1) Data processing & quality

- **Deduplicate aggressively:** add near-duplicate detection (e.g., cosine similarity on embeddings with a threshold ~0.95) before clustering to reduce topic dilution.
- **Thread handling:** when `include_thread_context=true`, bound context (e.g., previous 1–2 messages) and strip quoted histories/footers more deterministically (regex for `On .* wrote:` blocks and common signature delimiters).
- **Language detection:** optionally drop or separate non-English messages to avoid noisy clusters.

## 2) PII anonymization (tighten guarantees)

- **Deterministic salted hashing:** in Anonymizer, replace raw counters with `sha256(value + salt)[:8]` tokens (e.g., `EMAIL_<hash>`). Persist the salt per run so replacements are stable across batches without revealing originals.
- **Broaden patterns:** add IBAN/ACH/credit-card formats (masked), postal addresses (street suffix dictionary), invoice/PO patterns (`(INV|Invoice)[ -]?\d+`), and company names (gazetteer or spaCy NER pass).
- **Confidence tiers:** your `confidence_threshold` field suggests levels; route low-confidence matches to a “redact only” path rather than replacement, then re-scan before sending anything to the LLM.

## 3) Embedding generation

- **Model choice:** `all-MiniLM-L6-v2` is fast, but for nuanced business intents consider `all-mpnet-base-v2` (richer semantics) in discovery mode. Keep MiniLM for quick passes.
- **Batching & device:** lazy-load the model once, batch encode (e.g., 64/128), and expose `device` and `batch_size` in config.
- **Representation:** include a simple title (first line) + body dual-encoder style concat, capped lengths to reduce position bias.

## 4) Clustering & topic coverage

- **Representativeness:** don’t analyze only “top N” clusters by size; sample proportionally + include small, high-cohesion clusters (e.g., HDBSCAN with high probability scores).  
  Add a coverage report: % of emails assigned to a curated label vs “noise/unassigned”.
- **Auto labels (assist the LLM):** compute class-based TF-IDF keywords (c-TF-IDF) for each cluster to pass as hints to the LLM; or try BERTopic (UMAP + HDBSCAN + c-TF-IDF) purely for label proposals.

## 5) LLM analysis & prompting

- **Structured output:** require a strict JSON schema (intent, sentiment, definitions, examples, counter-examples, decision rules) and validate before saving. (This aligns with your Phase 2 artifact plan.)
- **Sampling per cluster:** pick the _k_ nearest to centroid and _k_ diverse outliers for each cluster (prevents bias). Keep temperature low (≤ 0.3). Seed prompts with current “working taxonomy” to encourage consolidation vs expansion.
- **Reliability:** add retries with exponential backoff and caching keyed by `(model, prompt_hash)` to control cost and make the pipeline resumable.
- **Guard PII:** re-anonymize after any transformations and before LLM calls; log a hash of inputs only.

## 6) Curation & taxonomy evolution

- **Decision rules:** for each category define: inclusion criteria, exclusion criteria, boundary examples, and a tie-breaker priority order. Make it machine-readable (YAML) so the classifier can use it at inference.
- **Versioning:** version the taxonomy (semver), store a `CHANGELOG.md`, and embed `taxonomy_version` in every downstream artifact and NetSuite note created.

## 7) Evaluation & Phase-2 classifier

- **Label a gold subset:** 150–250 messages, stratified across clusters. Store in `emails_labeled.csv` with intent, sentiment, and rationale.
- **Baselines:**
  - Heuristic baseline (keyword + regex rules derived from c-TF-IDF).
  - Shallow supervised baseline: logistic regression / linear SVM over frozen embeddings for intent; NB/SVM for sentiment.
- **LLM vs baselines:** produce per-class precision/recall/F1 and a confusion matrix. Track an “unassign” class to measure abstention. Target ≥ 85% precision on top intents.

## 8) Productionization & NetSuite fit

- **Hybrid inference:**
  1. Rules (fast wins: invoice keywords, payment verbs, request patterns).
  2. Small supervised model on embeddings for intent/sentiment.
  3. LLM fallback only when low confidence or OOD (include the taxonomy and decision rules in the prompt).
- **Confidence & rationale:** emit confidence, rationale, and modifiers (urgency/commitment/risk flags).
- **Observability:** structured JSON logs per message with the taxonomy version, model hash, and anonymizer salt id. Export to a BI table for drift checks.

## 9) Code quality & ops

- **Config:** migrate `PipelineConfig` to `pydantic-settings` for type-safe env + `.yaml` loading (env overrides per run).
- **Logging:** JSON logging with run id; keep plain `INFO` to console, full `DEBUG` to file. Add per-stage timers and cost accounting for LLM calls.
- **CLI:** ensure `run_pipeline.py` exposes all knobs and make runs idempotent (skip completed steps unless `--force`).
- **Tests:** quick unit tests with tiny fixtures: HTML cleaner, PII anonymizer (golden I/O), embedding determinism, clustering reproducibility (`random_state` pinned), and LLM schema validator (mocked).
- **Rate limiting:** analyzer should batch cluster analyses and respect OpenAI rate limits; add exponential backoff and jitter.
