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
- Successfully processes `litera_raw_emails.json` format
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

### 🔄 CURRENT FOCUS: Phase 2 Pipeline Improvements (PRIORITY)

**Issue Identified**: Current pipeline has accuracy limitations that need addressing before classifier development:
- Only 191 incoming emails from 4,732 total (4% rate seems too low)
- Limited sentiment diversity (only "Administrative" captured)
- Categories too narrow and administrative-focused
- Missing key collection-specific intents (payment disputes, hardship, etc.)

#### Phase 2.1: Email Direction Classification Enhancement ⚡ **IMMEDIATE PRIORITY**

1. **Analyze Current Classification Issues** 🔍
   - [ ] Review email direction classification accuracy in `data_processor.py`
   - [ ] Manually sample 50 emails marked as "outgoing" to verify correctness
   - [ ] Identify patterns in misclassified customer emails
   - [ ] Document specific cases where genuine customer emails are missed

2. **Enhance Collection-Specific Pattern Recognition** 🎯
   - [ ] Add collection-specific incoming patterns to `_classify_with_confidence()`:
     - Payment-related: "payment plan", "financial hardship", "dispute charge", "cannot pay"
     - Information requests: "account balance", "payment history", "invoice question"
     - Acknowledgments: "received invoice", "will pay by", "making payment"
   - [ ] Implement sentiment-aware classification (frustrated customers more likely incoming)
   - [ ] Add entity-based classification (mentions of amounts, dates, account numbers)

3. **Implement Multi-Factor Validation** 🔧
   - [ ] Create `CollectionEmailClassifier` class with confidence scoring
   - [ ] Add thread-context analysis (conversation flow patterns)
   - [ ] Implement business-logic validation (invoice -> customer response patterns)
   - [ ] Add length-based heuristics (very short emails likely fragments)

4. **Testing and Validation** ✅
   - [ ] Test enhanced classification on existing dataset
   - [ ] Target: Increase incoming email detection by 50-100% while maintaining precision
   - [ ] Create validation set of 100 manually labeled emails
   - [ ] Achieve >90% accuracy on validation set

#### Phase 2.2: Enhanced Clustering and Sentiment Detection 📊

5. **Multi-Dimensional Clustering Approach** 🎪
   - [ ] Implement hierarchical clustering in `clusterer.py`:
     - Stage 1: Broad intent categories (min_cluster_size=8)
     - Stage 2: Sentiment sub-clusters within each intent (min_cluster_size=3)
   - [ ] Adjust UMAP parameters for finer granularity:
     - `n_neighbors=10` (from 15), `min_dist=0.05` (from 0.1)
     - `n_components=30` (from 50), add `cluster_selection_epsilon=0.3`

6. **Collection-Specific Sentiment Analysis** 😊😠😐
   - [ ] Create `SentimentAnalyzer` class in new `pipeline/sentiment_analyzer.py`
   - [ ] Implement pattern-based sentiment detection:
     - **Frustrated**: "unacceptable", "ridiculous", "fed up", "still waiting"
     - **Cooperative**: "working with you", "happy to provide", "trying to resolve"
     - **Apologetic**: "apologize", "sorry for", "regret delay", "my fault"
     - **Urgent**: "need immediately", "asap", "time sensitive"
     - **Confused**: "don't understand", "unclear why", "please explain"
   - [ ] Add confidence scoring for each sentiment (0.0-1.0)
   - [ ] Integrate sentiment analysis into clustering pipeline

7. **Entity Extraction for Collections** 💰📅
   - [ ] Create `CollectionEntityExtractor` class in `pipeline/entity_extractor.py`
   - [ ] Extract key entities:
     - Payment amounts (`$X,XXX.XX`, `USD X`, etc.)
     - Payment dates (`by March 15`, `within 30 days`, etc.)
     - Invoice numbers (`INV-12345`, `Invoice #XXX`, etc.)
     - Account numbers (masked patterns)
     - Urgency indicators (`urgent`, `ASAP`, `immediately`)
   - [ ] Store extracted entities in email metadata for classification

#### Phase 2.3: Advanced LLM Analysis and Prompting 🤖

8. **Enhanced Prompt Engineering** ✍️
   - [ ] Rewrite LLM analysis prompts in `analyzer.py` with collection context:
     - Add collection business priorities (payment commitment, disputes, hardship)
     - Include specific intent categories (Payment Inquiry, Dispute, Arrangement Request)
     - Add sentiment context (frustrated customers, cooperative responses)
   - [ ] Implement few-shot prompting with real examples
   - [ ] Add structured output requirements (JSON schema enforcement)

9. **Context-Aware Thread Analysis** 🧵
   - [ ] Enhance `_analyze_thread_conversation_pattern()` in `data_processor.py`
   - [ ] Add conversation flow analysis:
     - Invoice → Customer Response → Resolution patterns
     - Escalation patterns (customer → manager → legal mentions)
     - Payment commitment tracking across thread
   - [ ] Implement thread-level sentiment progression analysis

10. **Business Logic Integration** 💼
    - [ ] Create `BusinessLogicValidator` class in `pipeline/business_validator.py`
    - [ ] Implement collection-specific validation rules:
      - Payment promises must have timeline/amount
      - Disputes must reference specific charges/invoices
      - Hardship claims should mention financial circumstances
    - [ ] Add actionability scoring for collection teams

#### Phase 2.4: Quality Metrics and Validation 📈

11. **Automated Quality Assessment** 🔬
    - [ ] Create `TaxonomyQualityAssessor` class in `pipeline/quality_assessor.py`
    - [ ] Implement quality metrics:
      - **Category Coverage**: Distribution across intent/sentiment categories
      - **Semantic Coherence**: Intra-cluster similarity vs inter-cluster distance
      - **Business Relevance**: Alignment with collection team priorities
      - **Actionability**: Clear next steps for each category
    - [ ] Generate quality reports with recommendations

12. **Validation Framework** ✅
    - [ ] Create ground truth dataset (200+ manually labeled emails)
    - [ ] Implement automated validation pipeline
    - [ ] Generate confusion matrices and performance metrics
    - [ ] Target metrics: >85% precision, >80% recall on top 5 categories

13. **A/B Testing Framework** 🧪
    - [ ] Implement parameter optimization testing
    - [ ] Compare current vs enhanced pipeline performance
    - [ ] Test different clustering parameters and prompt variations
    - [ ] Document performance improvements quantitatively

#### Implementation Priority Order

**Week 1**: Phase 2.1 (Email Direction Classification)
- Fix the core issue of missing customer emails
- This will immediately improve data quality for downstream analysis

**Week 2**: Phase 2.2 (Clustering and Sentiment)
- Enhanced clustering for better category discovery
- Multi-sentiment detection beyond just "Administrative"

**Week 3**: Phase 2.3 (LLM Enhancement)
- Improved prompts and business logic integration
- Context-aware analysis for better categorization

**Week 4**: Phase 2.4 (Quality and Validation)
- Automated quality assessment and validation framework
- Performance measurement and optimization

### 🎯 Phase 2 Success Criteria (Updated)

- [ ] **Data Quality**: Increase incoming email detection to 300-500 emails (vs current 191)
- [ ] **Sentiment Diversity**: Capture 4-6 distinct sentiment categories (vs current 1)
- [ ] **Intent Granularity**: Generate 6-8 actionable intent categories with business relevance
- [ ] **Classification Accuracy**: Achieve >85% precision on validation set
- [ ] **Business Utility**: Clear action items for collection teams for each category

### 🎯 Phase 3 - Prototype Classifier (After Pipeline Improvements)

#### Upcoming Tasks (Post-Improvements)

1. **JSON Schema Definition** 📋
   - [ ] Define strict JSON schema for model outputs
   - [ ] Include intent, tone, modifiers, entities, rationale fields
   - [ ] Validate schema against expected use cases

2. **Few-Shot Prompt Engineering** ✍️
   - [ ] Design few-shot prompts with system instructions
   - [ ] Create examples for each category using improved taxonomy
   - [ ] Test prompt effectiveness on enhanced dataset

3. **Manual Labeling for Validation** 🏷️
   - [ ] Select 150-250 email subset for ground truth (from improved dataset)
   - [ ] Manually label emails using enhanced taxonomy
   - [ ] Create validation dataset for model testing

4. **Model Performance Testing** 📈
   - [ ] Test LLM classifier on labeled subset
   - [ ] Generate confusion matrix analysis
   - [ ] Target ≥85% precision on top intent categories
   - [ ] Iterate and refine based on performance

#### Phase 3 Deliverables

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
│   └── litera_raw_emails.json  # Original data
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

**PRIORITY: Phase 2.1 Email Direction Classification Enhancement**

**Current Issue**: Only 191 incoming emails detected from 4,732 total (4% rate) - likely missing many genuine customer communications.

**Immediate Tasks**:
1. **Diagnostic Analysis** (Day 1-2):
   ```bash
   # Sample and manually review emails marked as outgoing
   uv run python -c "
   import json
   with open('outputs/output_analysis_1/processed_emails.json') as f:
       data = json.load(f)

   outgoing = [e for e in data['threads'] for email in e['emails'] if email['direction'] == 'outgoing']
   # Manually review sample of 50 emails to identify misclassifications
   "
   ```

2. **Enhanced Pattern Implementation** (Day 3-5):
   - Modify `pipeline/data_processor.py` `_classify_with_confidence()` method
   - Add collection-specific incoming patterns
   - Test on current dataset and measure improvement

3. **Validation and Testing** (Day 6-7):
   - Create validation set of 100 manually labeled emails
   - Test enhanced classification accuracy
   - Target: 300-500 incoming emails (2-3x improvement)

**Expected Outcome**:
- Significantly more comprehensive customer email dataset
- Better foundation for downstream clustering and taxonomy generation

**Pipeline Re-run Command** (after improvements):
```bash
# Test enhanced classification on current dataset
uv run python run_pipeline.py --input "Collection Notes - Sentiment Analysis/litera_raw_emails.json"

# Output will be in outputs/output_analysis_2/ with improved email detection
```

---

*This TODO will be updated as tasks are completed and new requirements emerge.*