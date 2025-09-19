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

### ✅ PHASE 1.5 COMPLETED: Reusable Pipeline Development

**Status**: COMPLETE - Full pipeline successfully implemented

**Key Accomplishments**:
- Built complete reusable pipeline system in `pipeline/` branch
- Successfully processes `CollectionNotes_SentimentAnalysis_SampleEmails.json` format
- Proper email thread separation (4,697 → 4,732 emails)
- Email direction classification (191 incoming, 4,541 outgoing)
- Complete taxonomy generation with rich formatting
- Generated production-ready `taxonomy.yaml` matching reference structure

**Pipeline Components**:
- `pipeline/data_processor.py` - Email parsing, HTML cleaning, thread separation
- `pipeline/anonymizer.py` - PII detection and anonymization
- `pipeline/embedder.py` - Vector embedding generation
- `pipeline/clusterer.py` - UMAP + HDBSCAN clustering
- `pipeline/analyzer.py` - GPT-4o cluster analysis
- `pipeline/curator.py` - Rich taxonomy generation and formatting
- `run_pipeline.py` - CLI entry point

### ✅ RESOLVED: Category Consolidation and Output Directory Standardization

**Issue**: Pipeline generated too many similar categories and inconsistent output naming

**Solution Implemented** (September 19, 2025):

#### 1. **Category Consolidation Improvements** ✅
- **Semantic Similarity Analysis**: Added sentence-transformer based similarity scoring with 0.92 threshold
- **Business Value Preservation**: Categories with distinct business purposes (payment/invoice/admin) preserved separately
- **Enhanced Consolidation Rules**: Pattern-based merging only for truly duplicate administrative variants
- **Result**: 7 duplicate categories → 4 meaningful business categories

#### 2. **Standardized Output Directory Naming** ✅
- **Auto-numbered Directories**: Implemented `output_analysis_1`, `output_analysis_2`, etc.
- **Incremental Logic**: Automatically finds next available number in outputs/ folder
- **CLI Updates**: Made `--dataset-name` optional, defaults to auto-numbering
- **Clean Structure**: Cleared legacy output directories for consistent naming

#### 3. **Final Category Quality** ✅
Latest pipeline run (`outputs/output_analysis_1/`) produces:
1. **Administrative Update** (75.9%) - Contact info, document confirmations
2. **Case and Invoice Follow-up** (14.1%) - Status updates, case closure
3. **Administrative Update and Payment Coordination** (7.3%) - Billing coordination
4. **Administrative Communication and Feedback** (2.6%) - Surveys, general admin

**Files Modified**:
- `pipeline/curator.py` - Added semantic similarity and business logic preservation
- `pipeline/config.py` - Implemented auto-numbered directory generation
- `run_pipeline.py` - Updated CLI for optional dataset naming
- `pipeline/pipeline.py` - Fixed summary generation for string taxonomy format

**Commit**: `03a23b7` - "Improve category consolidation and implement auto-numbered output directories"

**Key Implementation Features**:
- `_calculate_semantic_similarity()` - Uses sentence-transformers for category comparison
- `_has_distinct_business_value()` - Preserves payment/invoice/admin distinctions
- `_merge_similar_categories()` - High threshold (0.92) prevents over-merging
- `_apply_business_consolidation_rules()` - Selective pattern matching for duplicates
- `ConfigManager._get_next_analysis_number()` - Auto-generates output directory numbers

**Pipeline Status**: Ready for Phase 2 development with quality category generation

### 🎯 Phase 2 - Prototype Classifier (After Consolidation Fix)

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
├── taxonomy.yaml                               # Reference taxonomy (master branch)
├── Collection Notes - Sentiment Analysis/
│   └── CollectionNotes_SentimentAnalysis_SampleEmails.json  # Original data
│
├── pipeline/                                   # Reusable pipeline system
│   ├── __init__.py
│   ├── config.py                              # Pipeline configuration
│   ├── data_processor.py                      # Email parsing, thread separation
│   ├── anonymizer.py                          # PII detection and anonymization
│   ├── embedder.py                            # Vector embedding generation
│   ├── clusterer.py                           # UMAP + HDBSCAN clustering
│   ├── analyzer.py                            # GPT-4o cluster analysis
│   └── curator.py                             # Taxonomy generation ⚠️ NEEDS FIX
│
├── run_pipeline.py                             # CLI entry point
├── test_rich_curation.py                      # Testing script
│
├── outputs/                                   # Pipeline outputs
│   ├── collection_notes_test/                 # Initial test run
│   ├── collection_notes_rich/                 # Rich content test
│   └── collection_notes_final/                # Latest complete run ⚠️
│       ├── taxonomy.yaml                      # Generated taxonomy (7 duplicate categories)
│       ├── taxonomy_labeling_guide.md         # Comprehensive guide
│       ├── processed_emails.json              # 4,732 processed emails
│       ├── anonymized_emails.json             # PII-safe dataset
│       ├── cluster_results.json               # 7 clusters found
│       ├── taxonomy_analysis.json             # Rich LLM analysis
│       └── curation_summary.json              # Pipeline statistics
│
└── Legacy files (master branch):
    ├── master_email_threads.json              # Original processed dataset
    ├── master_email_threads_anonymized.json   # Original anonymized dataset
    ├── incoming_email_embeddings.npy          # Original embeddings
    ├── generate_embeddings_threaded.py        # Original scripts
    └── cluster analyses from Phase 1...
```

### ✅ Current Pipeline Status

**Latest Run**: `outputs/output_analysis_1/` (Auto-numbered)
- ✅ Successfully processed 4,732 emails from original 5,000 records
- ✅ Generated rich taxonomy.yaml with proper formatting
- ✅ Created comprehensive labeling guide
- ✅ **RESOLVED**: 4 meaningful business categories (was 7 duplicates)
- ✅ **IMPLEMENTED**: Auto-numbered output directories

**Categories Generated** (Meaningful Business Categories):
1. Administrative Update (75.9%) - Contact info, document confirmations
2. Case and Invoice Follow-up (14.1%) - Status updates, case closure
3. Administrative Update and Payment Coordination (7.3%) - Billing coordination
4. Administrative Communication and Feedback (2.6%) - Surveys, general admin

**Quality Achieved**:
- Distinct business value in each category
- Proper coverage distribution
- Clear decision rules and examples
- Ready for Phase 2 classifier development

### 🎯 Success Metrics

#### Phase 1 Success Criteria ✅ COMPLETED
- [x] Clear, mutually exclusive taxonomy categories
- [x] Good cluster separation in embedding space (5 clusters identified)
- [x] Domain expert validation of proposed categories (business-relevant categories)
- [x] Comprehensive labeling guide with examples

#### Phase 2 Success Criteria
- [ ] ≥85% precision on top intent categories
- [ ] Clear confusion matrix showing model performance
- [ ] Production-ready prompt and schema
- [ ] Validated approach for NetSuite integration

### 🚀 Next Immediate Action

**Phase 2: Prototype Classifier Development**
- Pipeline infrastructure complete with quality category generation
- Ready to begin classifier development using generated taxonomy
- Use `outputs/output_analysis_1/taxonomy.yaml` as the reference taxonomy
- Begin with JSON schema definition and prompt engineering

**Pipeline Usage**:
```bash
# Generate new analysis with auto-numbered directory
uv run python run_pipeline.py --input "Collection Notes - Sentiment Analysis/CollectionNotes_SentimentAnalysis_SampleEmails.json"

# Output will be in outputs/output_analysis_2/ (next available number)
```

---

*This TODO will be updated as tasks are completed and new requirements emerge.*