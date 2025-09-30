# Scripts

Utility scripts for running and managing the pipeline.

## Main Scripts

- **`run_pipeline.py`** - Main pipeline execution script
- **`pipeline_monitor.py`** - Monitor running pipeline processes
- **`kill_pipelines.sh`** - Stop running pipeline processes

## Usage

### Run Pipeline

```bash
# Run with test data
python scripts/run_pipeline.py --input source_data/test_data.json --dataset-name test_run

# Run with full dataset
python scripts/run_pipeline.py --input source_data/litera_raw_emails_v3_fixed.json --dataset-name production_v1

# Run with custom config
python scripts/run_pipeline.py --config config/my_config.yaml
```

### Monitor Processes

```bash
# View running pipeline processes
python scripts/pipeline_monitor.py
```

### Kill Processes

```bash
# Stop all running pipeline processes
./scripts/kill_pipelines.sh
```

## Requirements

All scripts require the Python environment to be activated:

```bash
# Using uv (recommended)
uv run python scripts/run_pipeline.py ...

# Or activate environment
source .venv/bin/activate
python scripts/run_pipeline.py ...
```
