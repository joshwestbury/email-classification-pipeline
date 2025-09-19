# Email Taxonomy Discovery Pipeline Documentation

## Pipeline Overview

The pipeline (`pipeline/pipeline.py`) orchestrates a 6-stage process to automatically discover sentiment and intent categories from customer collection emails. Each stage builds on the previous one, ultimately producing a curated taxonomy for NetSuite automation.

## Stage-by-Stage Breakdown

### Stage 1: Data Processing (`data_processor.py`)
**Purpose**: Clean and structure raw email data
- **HTML Cleaning**: Removes HTML tags, scripts, styles, and normalizes whitespace from email content
- **Thread Separation**: Parses email threads into individual messages using patterns like "On [date] wrote:" and "From:"
- **Direction Classification**: Determines if emails are incoming (from customers) or outgoing (from company) based on sender domains and content patterns
- **Output**: Structured emails with clean text, direction labels, and metadata

### Stage 2: PII Anonymization (`anonymizer.py`)
**Purpose**: Remove sensitive information while preserving business context
- **Pattern Detection**: Uses regex patterns to identify emails, phone numbers, addresses, account numbers, tax IDs
- **Consistent Replacement**: Generates consistent anonymized placeholders (e.g., `[EMAIL_1_A3B4C5]`) using MD5 hashing
- **Business Preservation**: Keeps @litera.com emails intact for direction classification
- **Output**: Anonymized dataset with PII replaced but business patterns preserved

### Stage 3: Embedding Generation (`embedder.py`)
**Purpose**: Convert text into numerical vectors for similarity analysis
- **Email Filtering**: Focuses only on incoming customer emails for taxonomy discovery
- **Text Preparation**: Combines subject and content for comprehensive embedding
- **Vector Generation**: Uses SentenceTransformer (all-MiniLM-L6-v2) to create 384-dimensional embeddings
- **Thread Context**: Optionally creates separate embeddings for full conversation threads
- **Output**: Numerical vectors representing semantic meaning of emails

### Stage 4: Clustering (`clusterer.py`)
**Purpose**: Group semantically similar emails together
- **Dimensionality Reduction**: Uses UMAP to reduce 384D embeddings to 50D for clustering efficiency
- **Clustering**: Applies HDBSCAN to identify natural groupings of similar emails
- **Quality Metrics**: Calculates silhouette score and noise ratios to assess cluster quality
- **Analysis**: Extracts sample emails from each cluster for LLM analysis
- **Output**: Cluster assignments with statistics and sample emails per cluster

### Stage 5: LLM Analysis (`analyzer.py`)
**Purpose**: Use AI to understand what each cluster represents
- **Sample Selection**: Picks representative emails from each cluster (top 8 clusters by size)
- **Prompt Engineering**: Sends structured prompts to GPT-4o asking for intent/sentiment categorization
- **Category Extraction**: Parses LLM responses to extract proposed categories with definitions and rules
- **Business Context**: Focuses specifically on collections/accounts receivable scenarios
- **Output**: Proposed intent/sentiment categories with definitions and business relevance

### Stage 6: Taxonomy Curation (`curator.py`)
**Purpose**: Refine LLM proposals into production-ready taxonomy
- **Semantic Merging**: Uses sentence transformers to merge categories with >92% similarity (high threshold to preserve business distinctions)
- **Business Rules**: Applies domain-specific consolidation while preserving important distinctions (e.g., payment vs invoice categories)
- **YAML Generation**: Creates structured taxonomy files with examples, decision rules, and NetSuite integration mappings
- **Documentation**: Generates comprehensive labeling guides for human annotators
- **Output**: Final taxonomy.yaml, labeling guide, and curation statistics

## Key Design Principles

1. **Customer-Focused**: Filters to incoming emails only since the goal is understanding customer communication patterns
2. **Business Awareness**: Preserves distinctions important for collections operations (payment vs invoice vs information requests)
3. **Production Ready**: Generates structured outputs ready for NetSuite integration
4. **Quality Control**: Each stage includes validation and metrics to assess processing quality
5. **Reusable**: Modular design allows processing new email datasets with consistent methodology

## Data Flow

The pipeline transforms raw email data through progressive refinement:

```
Raw Email Data → Clean Text → Mathematical Vectors → Semantic Clusters → Business Categories → Production Taxonomy
```

Each transformation adds value while preserving the essential business meaning needed for automated customer service systems.

## Usage

```bash
# Process new email dataset
python run_pipeline.py --input new_emails.json --dataset-name dataset_name

# Use custom configuration
python run_pipeline.py --config my_config.yaml
```

## Output Structure

```
outputs/{dataset_name}/
├── processed_emails.json     # Stage 1: Cleaned data
├── anonymized_emails.json    # Stage 2: PII-removed
├── embeddings/              # Stage 3: Vector representations
├── cluster_results.json     # Stage 4: Clustering analysis
├── taxonomy_analysis.json   # Stage 5: LLM proposals
├── taxonomy.yaml           # Stage 6: Final taxonomy
├── taxonomy_labeling_guide.md # Stage 6: Documentation
└── pipeline_summary.json    # Run summary and metrics
```