# AgenticBI Improvement Tasks

## Context
AgenticBI is a Vizro/Plotly Dash BI dashboard. The main file app/main.py is 1,927 lines and monolithic. We need to improve code quality, add documentation, and enhance UX.

## Task 1: Create Professional README.md
Replace the current README.md with a comprehensive one that includes:
- Project title and description
- Architecture overview (mentioning Vizro, Plotly Dash, Python backend)
- Key features list with emojis
- Setup instructions (git clone, pip install, env setup, run)
- Directory structure
- Screenshots placeholder section
- Contributing guidelines
- License (MIT)

## Task 2: Add pyproject.toml
Create a modern pyproject.toml with:
- [build-system] using setuptools
- [project] metadata (name="agenticbi", version="0.1.0", description, authors=[{name="Kushagra Kshatri", email="kushagrakshatri16@gmail.com"}])
- Dependencies from requirements.txt
- [project.optional-dependencies] dev = ["pytest", "black", "flake8", "mypy"]
- [tool.pytest.ini_options] testpaths = ["tests"]

## Task 3: Add Makefile
Create a Makefile with targets:
- install: pip install -e ".[dev]"
- test: pytest -v
- lint: black app/ tests/ && flake8 app/ tests/
- run: python app/main.py
- clean: remove __pycache__, .pyc files

## Task 4: Add tests/conftest.py
Create pytest fixtures for the Dash app.

## Task 5: Improve .env.example
Expand with all configuration options and comments.

Do NOT modify app/main.py or app/assets/custom.css — those will be handled separately.
