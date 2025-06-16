# Makefile for Bloom Quantization Project

# --- Variables ---
PYTHON = python3
VENV_DIR = venv
MODEL_DIR = bloom-560m-8bit

# Phony targets don't represent files
.PHONY: all venv install quantize verify test clean help

all: venv install test quantize verify

# --- Targets ---

venv:
	@echo ">>> Creating Python virtual environment in $(VENV_DIR)..."
	@$(PYTHON) -m venv $(VENV_DIR)
	@echo ">>> To activate, run: source $(VENV_DIR)/bin/activate"

install: venv
	@echo ">>> Installing dependencies from requirements.txt..."
	@$(VENV_DIR)/bin/pip install --upgrade pip
	@$(VENV_DIR)/bin/pip install transformers huggingface-hub safetensors psutil --no-deps
	@$(VENV_DIR)/bin/pip install pytest

test:
	@echo ">>> Running test suite..."
	@$(VENV_DIR)/bin/pytest test_quantize.py -v

quantize:
	@echo ">>> Running model quantization..."
	@$(VENV_DIR)/bin/python quantize_bloom.py --out_dir $(MODEL_DIR)

verify:
	@echo ">>> Verifying quantized model..."
	@$(VENV_DIR)/bin/python verify_quantized.py --model_dir $(MODEL_DIR)

clean:
	@echo ">>> Cleaning up generated files and cache..."
	@rm -rf $(VENV_DIR) $(MODEL_DIR) .cache __pycache__ *.pyc .pytest_cache
	@find . -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	@echo ">>> Done."

help:
	@echo "Available commands:"
	@echo "  make venv      - Sets up the Python virtual environment."
	@echo "  make install   - Installs all required dependencies."
	@echo "  make test      - Runs the test suite."
	@echo "  make quantize  - Runs the quantization script."
	@echo "  make verify    - Verifies the quantized model."
	@echo "  make all       - Runs venv, install, test, quantize, and verify."
	@echo "  make clean     - Removes all generated artifacts." 