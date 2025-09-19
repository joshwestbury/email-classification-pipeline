# Email Taxonomy Discovery Pipeline

A reusable pipeline for discovering sentiment and intent taxonomies from email datasets.

## Quick Start

1. **Create a configuration file:**
```bash
python run_pipeline.py --create-template
```

2. **Edit the template** (`pipeline_config_template.yaml`) with your dataset details

3. **Run the pipeline:**
```bash
python run_pipeline.py --config your_dataset.yaml
```

## Pipeline Steps

The pipeline performs these steps automatically:

1. **Data Processing** - Clean HTML, separate email threads
2. **PII Anonymization** - Remove sensitive information
3. **Embedding Generation** - Create vector representations
4. **Clustering** - Group similar emails using UMAP + HDBSCAN
5. **LLM Analysis** - Generate taxonomy categories using GPT-4o

## Configuration Options

Key parameters you can adjust:

- **Clustering sensitivity**: `hdbscan_min_cluster_size` (smaller = more clusters)
- **Embedding model**: `embedding_model` (try "all-mpnet-base-v2" for better quality)
- **Analysis depth**: `analyze_top_clusters` (more clusters = broader taxonomy)

## Output Files

For each run, you'll get:

- `processed_emails.json` - Cleaned and structured emails
- `anonymized_emails.json` - PII-removed dataset
- `embeddings/` - Vector representations
- `cluster_results.json` - Clustering analysis
- `taxonomy_analysis.json` - LLM-proposed categories
- `pipeline_summary.json` - Run summary and metrics

## Example Usage

**Process a new email dataset:**
```bash
# Quick run (uses defaults)
python run_pipeline.py --input new_emails.json --dataset-name customer_support

# Custom configuration
cp pipeline_config_template.yaml support_config.yaml
# Edit support_config.yaml with your settings
python run_pipeline.py --config support_config.yaml
```

**Experiment with different clustering:**
```yaml
# In your config file, try different values:
hdbscan_min_cluster_size: 3    # More granular clusters
umap_n_neighbors: 30           # Smoother embedding space
analyze_top_clusters: 12       # Analyze more clusters
```

## Requirements

- Python 3.8+
- OpenAI API key (set `OPENAI_API_KEY` environment variable)
- Input data in same format as `CollectionNotes_SentimentAnalysis_SampleEmails.json`

## Next Steps After Pipeline Run

1. Review the proposed taxonomy in `taxonomy_analysis.json`
2. Refine categories based on business needs
3. Use output for Phase 2: building the production classifier