# Configuration

Pipeline configuration templates and examples.

## Templates

- **`pipeline_config_template.yaml`** - Base configuration template with all options

## Usage

### Create Custom Configuration

```bash
# Copy template
cp config/pipeline_config_template.yaml my_config.yaml

# Edit configuration
vim my_config.yaml

# Run pipeline with config
python scripts/run_pipeline.py --config my_config.yaml
```

### Configuration Options

The template includes settings for:
- **Data Processing**: Anonymization, thread context
- **Embeddings**: Model selection, batch size
- **Clustering**: UMAP and HDBSCAN parameters
- **LLM Analysis**: Model, prompt settings
- **Output**: Save paths, intermediate files

### Common Configurations

**Quick Test Run** (faster, less accurate):
```yaml
embedding_model: "all-MiniLM-L6-v2"
umap_n_components: 30
min_cluster_size: 5
```

**Production Run** (slower, more accurate):
```yaml
embedding_model: "all-mpnet-base-v2"
umap_n_components: 50
min_cluster_size: 8
```

## Best Practices

1. **Keep template clean** - Don't modify `pipeline_config_template.yaml`
2. **Version your configs** - Save different configs for different datasets
3. **Document changes** - Add comments explaining custom settings
4. **Test first** - Validate config with small dataset before full run
