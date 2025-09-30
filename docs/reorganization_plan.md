# Repository Reorganization Plan

## Current Issues

1. **Root directory clutter**: 28+ files in root including tests, docs, configs, and scripts
2. **Mixed concerns**: Documentation, implementation notes, test files all at same level
3. **No clear structure**: Hard to find what you need quickly
4. **Test files scattered**: No dedicated test directory

---

## Proposed Structure

```
scg-ai-collection-notes/
├── README.md                          # Main project overview (keep)
├── CLAUDE.md                          # AI assistant instructions (keep)
├── pyproject.toml                     # Python project config (keep)
├── uv.lock                            # Dependency lock file (keep)
├── .gitignore                         # Git ignore rules (keep)
├── .env                               # Environment variables (keep)
├── .python-version                    # Python version (keep)
├── .anonymization_salt                # Security file (keep)
│
├── docs/                              # 📚 All documentation
│   ├── implementation/                # Implementation details
│   │   ├── phase_2_3_summary.md
│   │   ├── indicator_distinctiveness.md
│   │   ├── real_email_examples.md
│   │   └── process_management.md
│   ├── guides/                        # User guides
│   │   ├── pipeline_usage.md
│   │   └── taxonomy_labeling_guide.md
│   ├── legacy/                        # Old docs for reference
│   │   └── documentation.md
│   └── planning/                      # Planning docs
│       └── todo.md
│
├── pipeline/                          # ✅ Core pipeline code (already good)
│   ├── __init__.py
│   ├── anonymizer.py
│   ├── clusterer.py
│   ├── curator.py
│   ├── data_processor.py
│   ├── embedder.py
│   └── ...
│
├── scripts/                           # 🔧 Utility scripts
│   ├── run_pipeline.py                # Main pipeline runner
│   ├── pipeline_monitor.py            # Process monitoring
│   ├── kill_pipelines.sh              # Process management
│   └── README.md                      # Scripts documentation
│
├── tests/                             # 🧪 All test files
│   ├── __init__.py
│   ├── test_consolidation.py
│   ├── test_prompt_generator.py
│   ├── test_system_prompt.py
│   ├── test_validation_improvements.py
│   ├── fixtures/                      # Test data
│   │   ├── test_system_prompt.json
│   │   └── test_system_prompt.txt
│   └── README.md                      # Test documentation
│
├── config/                            # ⚙️ Configuration templates
│   ├── pipeline_config_template.yaml
│   └── README.md
│
├── source_data/                       # ✅ Raw input data (already good)
│   ├── litera_raw_emails_v3_fixed.json  # Latest production data
│   ├── test_data.json                   # Test subset
│   └── legacy/                          # Old versions
│       ├── litera_raw_emails.json
│       ├── litera_raw_emails_v2.json
│       └── litera_raw_emails_v3.json
│
├── outputs/                           # ✅ Pipeline outputs (already good)
│   ├── litera_test_data/
│   ├── litera_v6/
│   └── ...
│
├── artifacts/                         # 📦 Final deliverables
│   ├── taxonomy.yaml                  # Latest production taxonomy
│   ├── taxonomy_labeling_guide.md     # Latest guide
│   └── system_prompts/                # Production prompts
│       └── latest.txt
│
└── __pycache__/                       # Python cache (gitignored)
```

---

## Migration Script

Here's a bash script to reorganize everything safely:

```bash
#!/bin/bash
# reorganize.sh - Safely reorganize repository structure

set -e  # Exit on error

echo "🗂️  Reorganizing repository structure..."

# Create new directory structure
echo "Creating new directories..."
mkdir -p docs/implementation
mkdir -p docs/guides
mkdir -p docs/legacy
mkdir -p docs/planning
mkdir -p scripts
mkdir -p tests/fixtures
mkdir -p config
mkdir -p source_data/legacy
mkdir -p artifacts/system_prompts

# Move documentation files
echo "Moving documentation..."
mv PHASE_2_3_IMPLEMENTATION_SUMMARY.md docs/implementation/phase_2_3_summary.md
mv INDICATOR_DISTINCTIVENESS_IMPLEMENTATION.md docs/implementation/indicator_distinctiveness.md
mv REAL_EMAIL_EXAMPLES_IMPLEMENTATION.md docs/implementation/real_email_examples.md
mv PROCESS_MANAGEMENT.md docs/implementation/process_management.md
mv PIPELINE_README.md docs/guides/pipeline_usage.md
mv documentation.md docs/legacy/documentation.md
mv todo.md docs/planning/todo.md

# Move taxonomy artifacts to artifacts/
echo "Moving production artifacts..."
cp taxonomy.yaml artifacts/taxonomy.yaml
cp taxonomy_labeling_guide.md artifacts/taxonomy_labeling_guide.md

# Move scripts
echo "Moving scripts..."
mv run_pipeline.py scripts/
mv pipeline_monitor.py scripts/
mv kill_pipelines.sh scripts/

# Move test files
echo "Moving tests..."
mv test_*.py tests/
mv test_system_prompt.json tests/fixtures/
mv test_system_prompt.txt tests/fixtures/

# Move config
echo "Moving config..."
mv pipeline_config_template.yaml config/

# Move legacy source data
echo "Moving legacy source data..."
mv source_data/litera_raw_emails.json source_data/legacy/
mv source_data/litera_raw_emails_v2.json source_data/legacy/
mv source_data/litera_raw_emails_v3.json source_data/legacy/

# Create README files
echo "Creating README files..."

cat > scripts/README.md << 'EOF'
# Scripts

Utility scripts for running and managing the pipeline.

## Main Scripts

- `run_pipeline.py` - Main pipeline execution script
- `pipeline_monitor.py` - Monitor running pipeline processes
- `kill_pipelines.sh` - Stop running pipeline processes

## Usage

```bash
# Run pipeline
python scripts/run_pipeline.py --input source_data/test_data.json --dataset-name test_run

# Monitor processes
python scripts/pipeline_monitor.py

# Kill all pipelines
./scripts/kill_pipelines.sh
```
EOF

cat > tests/README.md << 'EOF'
# Tests

Unit and integration tests for the pipeline components.

## Running Tests

```bash
# Run all tests
pytest tests/

# Run specific test
pytest tests/test_consolidation.py

# Run with coverage
pytest --cov=pipeline tests/
```

## Test Files

- `test_consolidation.py` - LLM consolidation logic
- `test_prompt_generator.py` - System prompt generation
- `test_system_prompt.py` - Prompt validation
- `test_validation_improvements.py` - Validation logic
EOF

cat > config/README.md << 'EOF'
# Configuration

Pipeline configuration templates and examples.

## Templates

- `pipeline_config_template.yaml` - Base configuration template

## Usage

```bash
# Create custom config
cp config/pipeline_config_template.yaml my_config.yaml

# Edit configuration
vim my_config.yaml

# Run with config
python scripts/run_pipeline.py --config my_config.yaml
```
EOF

# Update .gitignore
echo "Updating .gitignore..."
cat >> .gitignore << 'EOF'

# Reorganization artifacts
docs/planning/todo.md
artifacts/*.yaml
artifacts/*.md
EOF

echo "✅ Reorganization complete!"
echo ""
echo "📋 Summary:"
echo "  - Moved documentation to docs/"
echo "  - Moved scripts to scripts/"
echo "  - Moved tests to tests/"
echo "  - Moved config to config/"
echo "  - Organized source_data/legacy/"
echo "  - Created artifacts/ for deliverables"
echo ""
echo "⚠️  Important: Run 'git status' to review changes before committing"
```

---

## Benefits of New Structure

### 1. **Clear Separation of Concerns**
- Documentation in `docs/`
- Code in `pipeline/`
- Tests in `tests/`
- Scripts in `scripts/`
- Config in `config/`

### 2. **Easier Navigation**
```bash
# Want to read implementation details?
cd docs/implementation/

# Want to run the pipeline?
cd scripts/

# Want to run tests?
cd tests/

# Want production artifacts?
cd artifacts/
```

### 3. **Better for New Team Members**
- Clear structure = faster onboarding
- README files in each directory
- Obvious where things belong

### 4. **Professional Appearance**
- Looks like a mature, well-organized project
- Ready for open-source or team collaboration
- Easy to create proper documentation site

### 5. **Cleaner Root Directory**
```
Before: 28+ files
After: ~10 files (core project files only)
```

---

## Migration Steps

### Step 1: Backup (Safety First!)
```bash
# Create a backup branch
git checkout -b backup-before-reorganization
git add .
git commit -m "Backup before reorganization"

# Create working branch
git checkout -b reorganize-structure
```

### Step 2: Run Migration Script
```bash
# Make script executable
chmod +x reorganize.sh

# Review the script
cat reorganize.sh

# Run it
./reorganize.sh
```

### Step 3: Update Import Paths
Some files will need updated import paths:

**Before**:
```python
from run_pipeline import main
```

**After**:
```python
from scripts.run_pipeline import main
```

Or add `scripts/` to Python path in key files.

### Step 4: Test Everything
```bash
# Test pipeline runs
python scripts/run_pipeline.py --help

# Test imports
python -c "from pipeline.curator import TaxonomyCurator; print('✅')"

# Run tests
cd tests && python test_consolidation.py
```

### Step 5: Update Documentation
Update these files with new paths:
- `README.md` - Update usage examples
- `CLAUDE.md` - Update file paths
- `docs/guides/pipeline_usage.md` - Update script paths

### Step 6: Commit
```bash
git add .
git commit -m "Reorganize repository structure for better maintainability

- Move documentation to docs/ directory
- Move scripts to scripts/ directory
- Move tests to tests/ directory
- Move config to config/ directory
- Create artifacts/ for production deliverables
- Organize source_data/legacy/ for old versions
- Add README files to new directories"
```

---

## Optional: Even More Organization

If you want to go further, consider:

### 1. **Documentation Site**
```bash
docs/
├── mkdocs.yml              # Documentation generator config
├── src/
│   ├── index.md           # Home page
│   ├── getting-started.md
│   ├── implementation/
│   └── guides/
```

Then use [MkDocs](https://www.mkdocs.org/) to generate a nice documentation site.

### 2. **GitHub Actions**
```bash
.github/
└── workflows/
    ├── tests.yml          # Run tests on push
    ├── docs.yml           # Deploy docs
    └── release.yml        # Create releases
```

### 3. **Pre-commit Hooks**
```bash
.pre-commit-config.yaml    # Code formatting, linting
```

---

## Alternative: Minimal Reorganization

If you want something less aggressive:

```
scg-ai-collection-notes/
├── README.md
├── CLAUDE.md
├── docs/                   # Just move all .md files here
├── scripts/                # Just move scripts here
├── tests/                  # Just move test files here
├── pipeline/               # Keep as is
├── source_data/            # Keep as is
└── outputs/                # Keep as is
```

This is simpler but still provides good organization.

---

## Recommendation

**Go with the full reorganization plan**. It's a one-time effort that will:
- Make the project more professional
- Easier to maintain long-term
- Better for collaboration
- Clearer for future you

The migration script handles the heavy lifting, and you can always revert if needed (you made a backup branch!).

---

## Questions to Consider

1. **Do you plan to share this repo?** → Full reorganization recommended
2. **Working solo forever?** → Minimal reorganization is fine
3. **Want CI/CD eventually?** → Full reorganization sets you up well
4. **Need quick access to outputs?** → Keep `outputs/` at root level

Let me know which approach you prefer and I can help with the migration!