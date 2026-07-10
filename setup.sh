#!/usr/bin/env bash
# Setup script for repository development
# Installs pre-commit and detect-secrets, and configures the git hook.

set -e

# Ensure pip is up‑to‑date
python -m pip install --upgrade pip

# Install development tools
python -m pip install --user pre-commit detect-secrets

# Install the pre‑commit hooks defined in .pre-commit-config.yaml
pre-commit install

# Optionally, create a baseline for detect‑secrets (run once)
# detect-secrets scan > .git/secrets.baseline

echo "Development environment setup complete."
