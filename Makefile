PYTHON ?= python3
VENV ?= .venv

.PHONY: install test app clean

install:
	$(PYTHON) -m venv $(VENV)
	$(VENV)/bin/python -m pip install --upgrade pip
	$(VENV)/bin/python -m pip install -r requirements.txt

test:
	$(VENV)/bin/python -m pytest -q

app:
	$(VENV)/bin/streamlit run app/streamlit_app.py

clean:
	rm -rf .pytest_cache
	find . -type d -name "__pycache__" -prune -exec rm -rf {} +

