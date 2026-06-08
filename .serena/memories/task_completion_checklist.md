# Task Completion Checklist

## Code Quality
- **Linting**: No formal linter configured (consider adding `pylint` or `flake8` if needed)
- **Testing**: No formal test suite found in the project
- **Type Checking**: Type hints are used but no `mypy` configuration found
- **Code Review**: Manual review recommended for prediction logic and data handling

## For Code Changes
1. Ensure type hints are added to new functions
2. Update docstrings/comments as needed
3. Test changes manually against the Streamlit UI (`./run.sh`)
4. Verify data integrity if modifying data pipeline
5. Test model predictions if modifying `model/` files

## Before Committing
1. Verify the app starts without errors: `./run.sh`
2. Check that predictions load and display correctly
3. Ensure no API keys are accidentally committed (already in `.env`)
4. Review changes don't break existing data/model loading

## Environment
- Python 3.9+ required
- `.env` file must exist with API keys
- macOS system (Darwin) with typical bash/zsh shell
