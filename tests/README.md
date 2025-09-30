# Tests

Unit and integration tests for the pipeline components.

## Test Files

- **`test_consolidation.py`** - LLM consolidation logic tests
- **`test_prompt_generator.py`** - System prompt generation tests
- **`test_prompt_on_refactored.py`** - Prompt validation after refactoring
- **`test_system_prompt.py`** - System prompt structure tests
- **`test_validation_improvements.py`** - Validation logic tests

## Test Fixtures

Test data and expected outputs are stored in `fixtures/`:
- `test_system_prompt.json` - Sample system prompt data
- `test_system_prompt.txt` - Sample system prompt text

## Running Tests

### Run All Tests

```bash
# From project root
python -m pytest tests/

# Or with uv
uv run pytest tests/
```

### Run Specific Test

```bash
python -m pytest tests/test_consolidation.py
```

### Run with Verbose Output

```bash
python -m pytest tests/ -v
```

### Run with Coverage

```bash
pytest --cov=pipeline tests/
```

## Adding New Tests

1. Create test file: `tests/test_your_feature.py`
2. Import modules: `from pipeline.your_module import YourClass`
3. Write tests following existing patterns
4. Add fixtures to `tests/fixtures/` if needed

## Test Conventions

- Test files must start with `test_`
- Test functions must start with `test_`
- Use descriptive test names
- Include docstrings explaining what is tested
