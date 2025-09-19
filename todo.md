# Collection Notes AI Analysis - Project TODO

## Project Status: Phase 1 - Taxonomy Discovery (In Progress)

### ✅ Completed Tasks

- [x] **Data Preparation and Cleaning**
  - [x] Clean up HTML tags from raw email JSON data (`clean_emails.py`)
  - [x] Extract clean text content while preserving structure
  - [x] Successfully processed 4,697 emails into `emails_cleaned.json`

- [x] **PII Detection and Anonymization**
  - [x] Create PII detection script (`detect_pii.py`)
  - [x] Identify emails containing sensitive information
  - [x] Create anonymization script (`anonymize_emails.py`)
  - [x] Anonymize addresses, emails, phone numbers, account numbers, tax IDs
  - [x] Generate anonymized dataset (`emails_anonymized.json`)

- [x] **Data Processing & Thread Separation** 🔧
  - [x] Parse and separate threaded email conversations into individual exchanges
  - [x] Clean HTML content and preserve only text content
  - [x] Classify email direction using @litera.com rule (outgoing vs incoming)
  - [x] Final dataset: 5,693 individual emails (703 incoming, 4,990 outgoing)
  - [x] Create master dataset (`master_email_threads.json`)

- [x] **PII Detection and Anonymization** 🔐
  - [x] Anonymize master email threads dataset (`anonymize_master_emails.py`)
  - [x] Preserve @litera.com emails for direction classification
  - [x] Anonymize emails, phone numbers, addresses, names, companies
  - [x] Generate anonymized dataset (`master_email_threads_anonymized.json`)
  - [x] Successfully anonymized 96.9% of emails (5,514/5,693)

- [x] **Generate Embeddings - Option A** 📊
  - [x] Install and set up embedding model (sentence-transformers)
  - [x] Implement Option A: Separate individual and thread context embeddings
  - [x] Create threaded embedding script (`generate_embeddings_threaded.py`)
  - [x] Choose appropriate embedding model for business/collection text (all-MiniLM-L6-v2)
  - [x] Generate individual incoming email embeddings (703 emails, 384 dims)
  - [x] Generate thread context embeddings (372 threads, 384 dims)
  - [x] Save embeddings and metadata (`incoming_email_embeddings.npy`, `thread_context_embeddings.npy`)
  - [x] **Option A Benefits**: Individual classification + contextual analysis capability

### ✅ PHASE 1 COMPLETED: Taxonomy Discovery

**Status**: COMPLETE - All deliverables generated and refined for production use

**Key Accomplishments**:
- Successfully clustered 703 incoming emails into 24 distinct groups
- Generated preliminary categories using GPT-4o analysis of top 8 clusters
- Refined taxonomy through human curation into production-ready categories
- Created comprehensive labeling guide with examples and decision rules

**Final Taxonomy**:
- **3 Intent Categories**: Payment Inquiry, Invoice Management, Information Request
- **4 Sentiment Categories**: Cooperative, Administrative, Informational, Frustrated
- **Coverage**: 52.3% of emails from analyzed clusters
- **Business Value**: Clear actionable categories for NetSuite Collection Notes

### 🔄 Current Phase: Phase 1 - Taxonomy Discovery (COMPLETED)

#### Next Steps (Priority Order)

1. **Clustering Analysis** ✅
   - [x] Install UMAP and HDBSCAN libraries
   - [x] Create clustering script using UMAP for dimensionality reduction
   - [x] Apply HDBSCAN for clustering similar emails
   - [x] Experiment with different clustering parameters
   - [x] Visualize clusters to understand email groupings

2. **LLM Category Proposal** ✅
   - [x] Set up LLM integration (OpenAI API or local model)
   - [x] Create script to analyze email clusters
   - [x] Generate initial category names and definitions
   - [x] Propose decision rules for each category
   - [x] Generated 8 preliminary intent categories, 4 sentiment categories

3. **Human Curation and Validation** ✅
   - [x] Review LLM-proposed categories for business relevance
   - [x] Ensure categories are mutually exclusive
   - [x] Refine category definitions based on domain expertise
   - [x] Create examples and counterexamples for each category
   - [x] Consolidated into 3 intent + 4 sentiment production-ready categories

#### Phase 1 Deliverables ✅

- [x] **`taxonomy_draft.json`** — Initial categories with examples
- [x] **`taxonomy.yaml`** — Curated taxonomy with definitions and rules
- [x] **`taxonomy_labeling_guide.md`** — Guide with examples and counterexamples
- [x] **`cluster_analyses_llm.json`** — Detailed LLM analysis of email clusters
- [x] **`cluster_analysis_summary.json`** — Statistical cluster analysis results

### 🎯 Phase 2 - Prototype Classifier (Future)

#### Upcoming Tasks

1. **JSON Schema Definition** 📋
   - [ ] Define strict JSON schema for model outputs
   - [ ] Include intent, tone, modifiers, entities, rationale fields
   - [ ] Validate schema against expected use cases

2. **Prompt Engineering** ✍️
   - [ ] Design few-shot prompts with system instructions
   - [ ] Create examples for each category
   - [ ] Test prompt effectiveness on sample data

3. **Manual Labeling for Validation** 🏷️
   - [ ] Select 150-250 email subset for ground truth
   - [ ] Manually label emails using developed taxonomy
   - [ ] Create validation dataset for model testing

4. **Model Performance Testing** 📈
   - [ ] Test LLM classifier on labeled subset
   - [ ] Generate confusion matrix analysis
   - [ ] Target ≥85% precision on top intent categories
   - [ ] Iterate and refine based on performance

#### Phase 2 Deliverables

- [ ] **`response_schema.json`** — Strict JSON schema for model output
- [ ] **`emails_labeled.csv`** — Ground-truth labeled subset
- [ ] **`confusion_matrix.png`** — Model vs human performance visualization
- [ ] **`system_prompt.txt`** — Final prompt for production use

### 🔧 Technical Setup Needed

#### Dependencies to Install
- [x] `sentence-transformers` - For generating embeddings
- [x] `umap-learn` - For dimensionality reduction
- [x] `hdbscan` - For clustering
- [x] `matplotlib`, `seaborn` - For visualization
- [x] `plotly` - For interactive visualizations
- [x] `numpy`, `tqdm` - Supporting libraries
- [x] `openai` or local LLM setup - For category generation
- [x] `python-dotenv` - For environment variable management
- [ ] `pydantic` - For JSON schema validation

#### Infrastructure Considerations
- [x] Determine if local or cloud-based LLM processing (OpenAI GPT-4o selected)
- [x] Set up API keys for external services (OpenAI API key configured)
- [ ] Plan for computational requirements (embeddings + clustering)
- [ ] Consider data storage for intermediate results

### 📁 Current Project Structure

```
scg-ai-collection-notes/
├── CLAUDE.md                                    # Project guidance
├── todo.md                                     # This file
├── Collection Notes - Sentiment Analysis/
│   └── CollectionNotes_SentimentAnalysis_SampleEmails.json  # Original data
│
├── master_email_threads.json                   # Processed dataset (5,693 individual emails)
├── master_email_threads_anonymized.json        # Anonymized final dataset
│
├── incoming_email_embeddings.npy               # Individual email embeddings (703)
├── thread_context_embeddings.npy               # Thread context embeddings (372)
├── incoming_email_metadata.json                # Individual email metadata
├── thread_context_metadata.json                # Thread context metadata
├── embeddings_config.json                      # Embedding generation config
│
├── generate_embeddings_threaded.py             # Option A embedding generation
├── anonymize_master_emails.py                  # Master dataset anonymization
├── clean_emails.py                            # HTML cleaning script
├── detect_pii.py                              # PII detection script
├── anonymize_emails.py                        # Original anonymization script
└── Legacy files:
    ├── emails_cleaned.json                     # Previous cleaned data
    ├── emails_anonymized.json                  # Previous anonymized data
    ├── generate_embeddings.py                  # Original embedding script
    └── cluster_emails.py                       # Previous clustering script
```

### 🎯 Success Metrics

#### Phase 1 Success Criteria
- [ ] Clear, mutually exclusive taxonomy categories
- [ ] Good cluster separation in embedding space
- [ ] Domain expert validation of proposed categories
- [ ] Comprehensive labeling guide with examples

#### Phase 2 Success Criteria
- [ ] ≥85% precision on top intent categories
- [ ] Clear confusion matrix showing model performance
- [ ] Production-ready prompt and schema
- [ ] Validated approach for NetSuite integration

### 🚀 Next Immediate Action

**Start with Step 1: Generate Embeddings**
- Install `sentence-transformers` using `uv add sentence-transformers`
- Create `generate_embeddings.py` script
- Process anonymized emails into vector representations
- Save embeddings for clustering analysis

---

*This TODO will be updated as tasks are completed and new requirements emerge.*